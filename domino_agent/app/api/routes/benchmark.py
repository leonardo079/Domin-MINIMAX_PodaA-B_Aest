"""
routes/benchmark.py — Endpoints de benchmark/torneo entre estrategias.

Endpoints:
  GET   /api/benchmark/matchups  → matchups estándar disponibles
  POST  /api/benchmark/run       → torneo completo con progreso via SSE
"""
import asyncio
import csv
import json
import os
import time
import uuid
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.schemas import BenchmarkRequest
from app.core.profiler import CostProfiler
from app.core.game_runner import play_full_game
from app.strategies import STRATEGIES

router = APIRouter()

RESULTS_DIR = Path("benchmark_results")

DEFAULT_TOURNAMENT = [
    ("minimax_m", "manhattan", "random",    "Minimax(Manhattan) vs Random"),
    ("astar_m",   "astar",     "random",    "A*(Manhattan) vs Random"),
    ("dist_cmp",  "manhattan", "euclidean", "Manhattan vs Euclidean"),
    ("hybrid_mm", "hybrid",    "manhattan", "Hybrid vs Minimax(Manhattan)"),
    ("hybrid_r",  "hybrid",    "random",    "Hybrid vs Random"),
]


def _run_matchup(tag: str, name_a: str, name_b: str, label: str, n_games: int) -> dict:
    """Ejecuta n_games partidas entre name_a y name_b; devuelve estadísticas."""
    wins = {0: 0, 1: 0, -1: 0}
    turns_list: list[int] = []
    score_advantage: list[int] = []
    combined_prof_a = CostProfiler(name_a)
    combined_prof_b = CostProfiler(name_b)

    for _ in range(n_games):
        prof_a = CostProfiler(name_a)
        prof_b = CostProfiler(name_b)
        sa = STRATEGIES[name_a](player=0)
        sb = STRATEGIES[name_b](player=1)
        sa.set_profiler(prof_a)
        sb.set_profiler(prof_b)

        winner, turns, pa, pb = play_full_game(sa, sb, prof_a, prof_b)

        wins[winner] = wins.get(winner, 0) + 1
        turns_list.append(turns)
        score_advantage.append(pb - pa)
        combined_prof_a.metrics.extend(prof_a.metrics)
        combined_prof_b.metrics.extend(prof_b.metrics)

    return {
        "tag": tag,
        "label": label,
        "agent_a": name_a,
        "agent_b": name_b,
        "n_games": n_games,
        "wins_a": wins[0],
        "wins_b": wins[1],
        "draws": wins[-1],
        "win_rate_a": round(wins[0] / n_games * 100, 1),
        "win_rate_b": round(wins[1] / n_games * 100, 1),
        "avg_turns": round(sum(turns_list) / len(turns_list), 2),
        "turns_per_game": turns_list,
        "score_advantage_per_game": score_advantage,
        "metrics_a": combined_prof_a.summary(),
        "metrics_b": combined_prof_b.summary(),
    }


def _export_benchmark_csv(results: list[dict], run_id: str) -> dict[str, str]:
    """
    Exporta los resultados del benchmark en tres archivos CSV
    optimizados para pgfplots en LaTeX/Overleaf.

    Archivos generados en benchmark_results/:
      1. winrates_<run_id>.csv     — win rates por matchup
      2. avg_metrics_<run_id>.csv  — métricas promedio por agente y matchup
      3. turns_dist_<run_id>.csv   — duración y ventaja por partida
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    paths = {}

    # ── 1. Win rates por matchup ──────────────────────────────────────────
    path_wr = RESULTS_DIR / f"winrates_{run_id}.csv"
    with open(path_wr, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "matchup", "agent_a", "agent_b",
            "wins_a", "wins_b", "draws",
            "win_rate_a", "win_rate_b", "n_games",
        ])
        for r in results:
            writer.writerow([
                r["label"],
                r["agent_a"],
                r["agent_b"],
                r["wins_a"],
                r["wins_b"],
                r["draws"],
                r["win_rate_a"],
                r["win_rate_b"],
                r["n_games"],
            ])
    paths["winrates"] = str(path_wr)

    # ── 2. Métricas promedio por agente y matchup ─────────────────────────
    path_mt = RESULTS_DIR / f"avg_metrics_{run_id}.csv"
    with open(path_mt, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "matchup", "agent", "role",
            "avg_time_ms", "avg_nodes", "avg_evals",
            "avg_depth", "total_turns",
        ])
        for r in results:
            for role, key in [("A", "metrics_a"), ("B", "metrics_b")]:
                m = r[key]
                if not m:
                    continue
                agent_name = r["agent_a"] if role == "A" else r["agent_b"]
                writer.writerow([
                    r["label"],
                    agent_name,
                    role,
                    round(m.get("avg_time_ms", 0), 4),
                    round(m.get("avg_nodes", 0), 2),
                    round(m.get("avg_evals", 0), 2),
                    round(m.get("avg_depth", 0), 2),
                    m.get("turns", 0),
                ])
    paths["avg_metrics"] = str(path_mt)

    # ── 3. Distribución de duración de partidas ───────────────────────────
    path_td = RESULTS_DIR / f"turns_dist_{run_id}.csv"
    with open(path_td, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["matchup", "agent_a", "agent_b", "game_index", "turns", "score_advantage"])
        for r in results:
            for i, (t, s) in enumerate(
                zip(r["turns_per_game"], r["score_advantage_per_game"])
            ):
                writer.writerow([
                    r["label"],
                    r["agent_a"],
                    r["agent_b"],
                    i + 1,
                    t,
                    s,
                ])
    paths["turns_dist"] = str(path_td)

    return paths


# ── GET matchups estándar ──────────────────────────────────────────────────────

@router.get("/matchups")
def get_default_matchups():
    return {
        "matchups": [
            {"tag": tag, "agent_a": a, "agent_b": b, "label": label}
            for tag, a, b, label in DEFAULT_TOURNAMENT
        ]
    }


# ── POST /run — Torneo con streaming de progreso via SSE ──────────────────────

@router.post("/run")
async def run_benchmark(request: BenchmarkRequest):
    """
    Ejecuta el torneo y emite eventos SSE de progreso.
    Cada matchup emite un evento cuando termina.
    Al final emite el resumen completo y exporta los CSVs.
    """
    matchups = request.matchups or [
        {"tag": tag, "agent_a": a, "agent_b": b, "label": label}
        for tag, a, b, label in DEFAULT_TOURNAMENT
    ]
    n_games = request.n_games
    run_id = str(uuid.uuid4())[:8]

    async def event_generator():
        loop = asyncio.get_event_loop()
        t0 = time.time()
        all_results = []

        yield f"data: {json.dumps({'type': 'start', 'run_id': run_id, 'n_matchups': len(matchups), 'n_games': n_games})}\n\n"

        for idx, m in enumerate(matchups):
            tag = m.get("tag", f"matchup_{idx}")
            na = m["agent_a"]
            nb = m["agent_b"]
            label = m.get("label", f"{na} vs {nb}")

            yield f"data: {json.dumps({'type': 'matchup_start', 'index': idx, 'tag': tag, 'label': label})}\n\n"

            result = await loop.run_in_executor(
                None, _run_matchup, tag, na, nb, label, n_games
            )
            all_results.append(result)

            yield f"data: {json.dumps({'type': 'matchup_done', 'index': idx, **result})}\n\n"

        total_time = round(time.time() - t0, 2)

        # Exportar CSVs al terminar el torneo
        exported_paths = _export_benchmark_csv(all_results, run_id)

        yield f"data: {json.dumps({'type': 'benchmark_done', 'run_id': run_id, 'total_time_s': total_time, 'results': all_results, 'exported_files': exported_paths})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )