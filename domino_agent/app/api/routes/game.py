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

from app.api.schemas import NewGameRequest
from app.api import game_manager

router = APIRouter()


# ── Crear sesión ───────────────────────────────────────────────────────────────

@router.post("/new")
def new_game(request: NewGameRequest):
    session = game_manager.create_session(
        request.strategy_a.value,
        request.strategy_b.value,
    )
    return {
        "session_id": session.session_id,
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
    event = session.step()
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
