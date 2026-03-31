"""
routes/benchmark.py — Endpoints de benchmark/torneo entre estrategias.

Endpoints:
  GET   /api/benchmark/matchups  → matchups estándar disponibles
  POST  /api/benchmark/run       → torneo completo con progreso via SSE
"""
import asyncio
import json
import time
import uuid

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.schemas import BenchmarkRequest
from app.core.profiler import CostProfiler
from app.core.game_runner import play_full_game
from app.strategies import STRATEGIES

router = APIRouter()

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
        yield f"data: {json.dumps({'type': 'benchmark_done', 'run_id': run_id, 'total_time_s': total_time, 'results': all_results})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )