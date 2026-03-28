"""
routes/game.py — Endpoints de sesión de juego.

Endpoints:
  POST   /api/game/new              → crear sesión
  GET    /api/game/                 → listar sesiones
  GET    /api/game/{id}             → estado actual
  POST   /api/game/{id}/step        → ejecutar un turno
  GET    /api/game/{id}/stream      → SSE: auto-jugar y emitir eventos en tiempo real
  GET    /api/game/{id}/history     → historial completo de turnos
  DELETE /api/game/{id}             → eliminar sesión
"""
import asyncio
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.api.schemas import NewGameRequest, HumanMoveRequest
from app.api import game_manager

router = APIRouter()


# ── Crear sesión ───────────────────────────────────────────────────────────────

@router.post("/new")
def new_game(request: NewGameRequest):
    strategy_b = request.strategy_b.value if request.strategy_b else None
    session = game_manager.create_session(
        request.strategy_a.value,
        strategy_b,
        request.game_mode.value,
    )
    return {
        "session_id": session.session_id,
        "game_mode": session.game_mode,
        "strategy_a": session.strategy_a_name,
        "strategy_b": session.strategy_b_name,
        "status": session.status,
        "initial_state": session.get_state_snapshot(),
    }


# ── Listar sesiones ────────────────────────────────────────────────────────────

@router.get("/")
def list_games():
    return {"sessions": game_manager.list_sessions()}


# ── Estado actual ──────────────────────────────────────────────────────────────

@router.get("/{session_id}")
def get_game(session_id: str):
    session = game_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    return session.get_state_snapshot()


# ── Ejecutar un turno ──────────────────────────────────────────────────────────

@router.post("/{session_id}/step")
def step_game(session_id: str):
    session = game_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    if session.status == "finished":
        raise HTTPException(status_code=400, detail="La partida ya terminó")
    if session.status == "waiting_human":
        raise HTTPException(status_code=400, detail="Es el turno del humano — usa POST /human-move")
    try:
        event = session.step()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return event


# ── Turno del humano ───────────────────────────────────────────────────────────

@router.post("/{session_id}/human-move")
def human_move(session_id: str, request: HumanMoveRequest):
    """
    El humano envía su jugada. Solo válido en modo agent_vs_human
    cuando status == 'waiting_human'.

    Body: { "tile_a": 3, "tile_b": 5, "side": "right" }

    Responde con el evento del turno y el estado actualizado,
    incluyendo las jugadas válidas del humano si aún es su turno.
    """
    session = game_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    try:
        event = session.human_move(request.tile_a, request.tile_b, request.side)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return event


@router.post("/{session_id}/human-pass")
def human_pass(session_id: str):
    """
    El humano pasa su turno. Solo permitido cuando no hay jugadas
    válidas ni fichas en el pozo.
    """
    session = game_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    try:
        event = session.human_pass()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return event


# ── SSE: auto-jugar con eventos en tiempo real ─────────────────────────────────

@router.get("/{session_id}/stream")
async def stream_game(session_id: str, delay_ms: float = 0):
    """
    Server-Sent Events: auto-juega la partida turno a turno y emite un evento
    JSON por turno. El cliente puede mostrar gráficas en tiempo real con estos datos.

    Cada evento tiene el formato:
        data: { "type": "turn", "turn": N, "metrics": { ... }, ... }

    Al final:
        data: { "type": "game_over", "winner": 0|1|-1, "summary_a": {...}, ... }

    Query param:
        delay_ms (float, default=0): milisegundos de pausa entre turnos.
                                     Permite ver el juego a velocidad reducida.
    """
    session = game_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")

    async def event_generator():
        loop = asyncio.get_event_loop()
        while session.status != "finished":
            # Ejecutar turno en executor para no bloquear el event loop
            event = await loop.run_in_executor(None, session.step)
            yield f"data: {json.dumps(event)}\n\n"

            if event.get("is_terminal") or event.get("type") == "game_over":
                # Emitir resumen final
                summary = session._build_game_over_event()
                yield f"data: {json.dumps(summary)}\n\n"
                break

            if delay_ms > 0:
                await asyncio.sleep(delay_ms / 1000.0)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Árbol de búsqueda del último turno ────────────────────────────────────────

@router.get("/{session_id}/tree")
def get_tree(session_id: str):
    """
    Retorna el árbol de búsqueda generado por cada agente en su último turno.

    Estructura de respuesta:
      tree_a — árbol del agente A (siempre IA)
      tree_b — árbol del agente B (IA en agent_vs_agent, null si jugó el humano)

    Cada árbol contiene una lista de nodos con:
      id, parent_id, depth, node_type, move, alpha, beta, value, pruned

    node_type valores:
      "ROOT"          — raíz de decisión
      "MAX"/"MIN"     — nodos Minimax (α-β en Manhattan / Euclidean / Hybrid)
      "PRUNED"        — marcador de ramas cortadas por α-β
      "ASTAR"         — nodo en búsqueda A* (alpha=f, beta=h, value=g)
      "ASTAR_PHASE"   — fase de ranking A* en Hybrid
      "ASTAR_RANK"    — candidato clasificado en la fase A* del Hybrid
      "MINIMAX_PHASE" — fase Minimax α-β en Hybrid
      "RANDOM"        — jugada disponible en RandomStrategy
    """
    session = game_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    return session.get_last_trees()


# ── Historial de turnos ────────────────────────────────────────────────────────

@router.get("/{session_id}/history")
def get_history(session_id: str):
    session = game_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    return {
        "session_id": session_id,
        "turn_count": session.turn,
        "status": session.status,
        "history": session.turn_history,
    }


# ── Eliminar sesión ────────────────────────────────────────────────────────────

@router.delete("/{session_id}")
def delete_game(session_id: str):
    deleted = game_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    return {"deleted": session_id}
