"""
routes/metrics.py — Endpoints de métricas y análisis.

Endpoints:
  GET  /api/metrics/strategies              → descripción de estrategias disponibles
  GET  /api/metrics/game/{session_id}       → métricas completas de una sesión
  GET  /api/metrics/game/{session_id}/realtime  → datos formateados para gráficas en tiempo real
  GET  /api/metrics/game/{session_id}/summary   → resumen post-partida para gráficas finales

Nota sobre qué graficar:
  - TIEMPO REAL (por turno, durante la partida):
      time_ms, nodes_expanded, eval_calls, max_depth, hand_size, board_length
  - POST-PARTIDA (al finalizar):
      win_rate, score_advantage_convergence, turn_distribution, cumulative_costs, radar_comparison
"""
from fastapi import APIRouter, HTTPException

from app.api import game_manager
from app.strategies import STRATEGIES, STRATEGY_DESCRIPTIONS

router = APIRouter()


@router.get("/strategies")
def list_strategies():
    """Lista todas las estrategias disponibles con sus descripciones."""
    return {
        "strategies": [
            {"name": name, "description": desc}
            for name, desc in STRATEGY_DESCRIPTIONS.items()
        ]
    }


@router.get("/game/{session_id}")
def get_full_metrics(session_id: str):
    """
    Retorna todas las métricas recolectadas durante la sesión:
    - Métricas turno a turno de ambas estrategias (para gráficas)
    - Resúmenes agregados
    """
    session = game_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    return session.get_metrics_history()


@router.get("/game/{session_id}/realtime")
def get_realtime_chart_data(session_id: str):
    """
    Datos pre-formateados para las gráficas EN TIEMPO REAL del frontend.

    Estas gráficas tienen VALOR MÁXIMO durante la partida porque muestran
    el comportamiento instantáneo de cada algoritmo turno a turno:

    1. time_ms_per_turn     → ¿Cuánto tarda cada algoritmo en pensar?
    2. nodes_per_turn       → ¿Cuántos nodos expande por turno?
    3. evals_per_turn       → ¿Cuántas veces llama a la heurística?
    4. depth_per_turn       → ¿Qué tan profundo busca en el árbol?
    5. hand_sizes           → Tamaño de mano (progreso del juego)
    6. board_length         → Fichas en el tablero (avance del juego)

    Formato: series listas para Chart.js / Recharts / D3
    """
    session = game_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")

    turns_a = [m for m in session.turn_history if m.get("player") == 0 and m.get("metrics")]
    turns_b = [m for m in session.turn_history if m.get("player") == 1 and m.get("metrics")]

    def extract_series(turns: list, key: str) -> list:
        return [{"turn": t["turn"], "value": t["metrics"][key]} for t in turns if t.get("metrics")]

    return {
        "session_id": session_id,
        "strategy_a": session.strategy_a_name,
        "strategy_b": session.strategy_b_name,
        "current_turn": session.turn,
        "status": session.status,

        # ─── Métricas de costo computacional por turno ─────────────────
        "realtime_charts": {
            "time_ms": {
                "description": "Tiempo de decisión por turno (ms)",
                "chart_type": "line",
                "update": "per_turn",
                "series_a": extract_series(turns_a, "time_ms"),
                "series_b": extract_series(turns_b, "time_ms"),
            },
            "nodes_expanded": {
                "description": "Nodos expandidos por turno (espacio de búsqueda)",
                "chart_type": "line",
                "update": "per_turn",
                "series_a": extract_series(turns_a, "nodes_expanded"),
                "series_b": extract_series(turns_b, "nodes_expanded"),
            },
            "eval_calls": {
                "description": "Llamadas a la función heurística por turno",
                "chart_type": "bar",
                "update": "per_turn",
                "series_a": extract_series(turns_a, "eval_calls"),
                "series_b": extract_series(turns_b, "eval_calls"),
            },
            "max_depth": {
                "description": "Profundidad máxima de búsqueda por turno",
                "chart_type": "step",
                "update": "per_turn",
                "series_a": extract_series(turns_a, "max_depth"),
                "series_b": extract_series(turns_b, "max_depth"),
            },
        },

        # ─── Estado del juego por turno ────────────────────────────────
        "game_progress_charts": {
            "hand_sizes": {
                "description": "Fichas en mano por turno (progreso del juego)",
                "chart_type": "line",
                "update": "per_turn",
                "series_a": [
                    {"turn": t["turn"], "value": t["hand_size_a"]}
                    for t in session.turn_history
                ],
                "series_b": [
                    {"turn": t["turn"], "value": t["hand_size_b"]}
                    for t in session.turn_history
                ],
            },
            "board_length": {
                "description": "Longitud del tablero por turno",
                "chart_type": "area",
                "update": "per_turn",
                "series": [
                    {"turn": t["turn"], "value": t["board_length"]}
                    for t in session.turn_history
                ],
            },
            "pool_size": {
                "description": "Fichas restantes en el pozo",
                "chart_type": "area",
                "update": "per_turn",
                "series": [
                    {"turn": t["turn"], "value": t["pool_size"]}
                    for t in session.turn_history
                ],
            },
        },
    }


@router.get("/game/{session_id}/summary")
def get_endgame_chart_data(session_id: str):
    """
    Datos pre-formateados para las gráficas POST-PARTIDA del frontend.

    Estas gráficas son más útiles AL FINAL porque requieren datos agregados:

    1. cost_comparison   → Barras comparando avg_time, avg_nodes, avg_evals por estrategia
    2. cumulative_cost   → Costo acumulado durante la partida (tiempo total, nodos totales)
    3. depth_distribution→ Distribución de profundidades alcanzadas (histograma)
    4. radar             → Radar/spider chart de métricas normalizadas
    5. pip_balance       → Suma de pips finales de cada jugador
    """
    session = game_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")

    summary_a = session.prof_a.summary()
    summary_b = session.prof_b.summary()
    metrics_a = session.prof_a.all_metrics_list()
    metrics_b = session.prof_b.all_metrics_list()

    # Costo acumulado por turno
    def cumulative(metrics: list, key: str) -> list:
        total = 0.0
        result = []
        for m in metrics:
            total += m[key]
            result.append({"turn": m["turn"], "value": round(total, 3)})
        return result

    # Distribución de profundidad
    def depth_histogram(metrics: list) -> dict:
        from collections import Counter
        counts = Counter(m["max_depth"] for m in metrics)
        return {str(k): v for k, v in sorted(counts.items())}

    return {
        "session_id": session_id,
        "strategy_a": session.strategy_a_name,
        "strategy_b": session.strategy_b_name,
        "status": session.status,
        "winner": session.winner_id,
        "total_turns": session.turn,
        "pip_sum_a": session.state.pip_sum(0),
        "pip_sum_b": session.state.pip_sum(1),

        # ─── Gráficas post-partida ─────────────────────────────────────
        "endgame_charts": {
            "cost_comparison": {
                "description": "Comparación de costos promedio entre estrategias",
                "chart_type": "grouped_bar",
                "metrics": ["avg_time_ms", "avg_nodes", "avg_evals", "avg_depth"],
                "strategy_a": {k: summary_a.get(k, 0) for k in ["avg_time_ms", "avg_nodes", "avg_evals", "avg_depth"]},
                "strategy_b": {k: summary_b.get(k, 0) for k in ["avg_time_ms", "avg_nodes", "avg_evals", "avg_depth"]},
            },
            "cumulative_time_ms": {
                "description": "Tiempo de cómputo acumulado durante la partida",
                "chart_type": "area",
                "series_a": cumulative(metrics_a, "time_ms"),
                "series_b": cumulative(metrics_b, "time_ms"),
            },
            "cumulative_nodes": {
                "description": "Nodos expandidos en total durante la partida",
                "chart_type": "area",
                "series_a": cumulative(metrics_a, "nodes_expanded"),
                "series_b": cumulative(metrics_b, "nodes_expanded"),
            },
            "depth_distribution": {
                "description": "Distribución de profundidades alcanzadas",
                "chart_type": "histogram",
                "strategy_a": depth_histogram(metrics_a),
                "strategy_b": depth_histogram(metrics_b),
            },
            "radar": {
                "description": "Comparación multidimensional normalizada (menor = mejor)",
                "chart_type": "radar",
                "axes": ["avg_time_ms", "avg_nodes", "avg_evals", "avg_depth", "total_turns"],
                "strategy_a": {
                    "avg_time_ms": summary_a.get("avg_time_ms", 0),
                    "avg_nodes": summary_a.get("avg_nodes", 0),
                    "avg_evals": summary_a.get("avg_evals", 0),
                    "avg_depth": summary_a.get("avg_depth", 0),
                    "total_turns": summary_a.get("turns", 0),
                },
                "strategy_b": {
                    "avg_time_ms": summary_b.get("avg_time_ms", 0),
                    "avg_nodes": summary_b.get("avg_nodes", 0),
                    "avg_evals": summary_b.get("avg_evals", 0),
                    "avg_depth": summary_b.get("avg_depth", 0),
                    "total_turns": summary_b.get("turns", 0),
                },
            },
            "pip_balance": {
                "description": "Pips en mano al final (menor = mejor)",
                "chart_type": "bar",
                "strategy_a": session.state.pip_sum(0),
                "strategy_b": session.state.pip_sum(1),
            },
        },
    }
