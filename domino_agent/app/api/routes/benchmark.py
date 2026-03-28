"""
routes/benchmark.py — Endpoints de benchmark/torneo entre estrategias.

Endpoints:
  POST  /api/benchmark/run          → ejecutar torneo (bloqueante, progreso via SSE)
  GET   /api/benchmark/matchups     → matchups estándar disponibles
"""
import asyncio
import json
import time
import uuid
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.schemas import BenchmarkRequest
from app.core.game_state import GameState
from app.core.profiler import CostProfiler
from app.strategies import STRATEGIES

router = APIRouter()

DEFAULT_TOURNAMENT = [
    ("minimax_m", "manhattan", "random",    "Minimax(Manhattan) vs Random"),
    ("astar_m",   "astar",     "random",    "A*(Manhattan) vs Random"),
    ("dist_cmp",  "manhattan", "euclidean", "Manhattan vs Euclidean"),
    ("hybrid_mm", "hybrid",    "manhattan", "Hybrid vs Minimax(Manhattan)"),
    ("hybrid_r",  "hybrid",    "random",    "Hybrid vs Random"),
]


def _play_single_game(strategy_a: str, strategy_b: str) -> tuple:
    """Juega una partida y retorna (winner, turns, pips_a, pips_b)."""
    state = GameState.new_game()
    prof_a = CostProfiler(strategy_a)
    prof_b = CostProfiler(strategy_b)
    sa = STRATEGIES[strategy_a](player=0)
    sb = STRATEGIES[strategy_b](player=1)
    sa.set_profiler(prof_a)
    sb.set_profiler(prof_b)
    turn = 0

    while not state.is_terminal():
        turn += 1
        cur = state.current_player
        strat = sa if cur == 0 else sb
        hand = state.agent_hand if cur == 0 else state.opponent_hand
        moves = state.valid_moves(hand)

        if not moves and state.pool:
            state, moves = state.apply_draw_and_play(cur)

        result = strat.decide(state)
        if result is None:
            state = state.apply_pass(cur)
        else:
            tile, side = result
            state = state.apply_move(tile, side, cur)

    return state.winner(), turn, state.pip_sum(0), state.pip_sum(1), prof_a, prof_b


def _run_matchup(tag: str, name_a: str, name_b: str, label: str, n_games: int) -> dict:
    wins = {0: 0, 1: 0, -1: 0}
    turns_list = []
    score_advantage = []
    combined_prof_a = CostProfiler(name_a)
    combined_prof_b = CostProfiler(name_b)

    for _ in range(n_games):
        winner, turns, pa, pb, prof_a, prof_b = _play_single_game(name_a, name_b)
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
    Al final emite el resumen completo.

    Esto permite al frontend mostrar resultados parciales mientras corre.
    """
    matchups = request.matchups
    if not matchups:
        matchups = [
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

            # Notificar inicio de matchup
            yield f"data: {json.dumps({'type': 'matchup_start', 'index': idx, 'tag': tag, 'label': label})}\n\n"

            # Correr el matchup en executor (bloqueante, CPU intensivo)
            result = await loop.run_in_executor(
                None, _run_matchup, tag, na, nb, label, n_games
            )
            all_results.append(result)

            # Emitir resultado del matchup
            yield f"data: {json.dumps({'type': 'matchup_done', 'index': idx, **result})}\n\n"

        total_time = round(time.time() - t0, 2)
        yield f"data: {json.dumps({'type': 'benchmark_done', 'run_id': run_id, 'total_time_s': total_time, 'results': all_results})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
