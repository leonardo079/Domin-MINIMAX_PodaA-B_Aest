from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class StrategyName(str, Enum):
    random = "random"
    manhattan = "manhattan"
    euclidean = "euclidean"
    astar = "astar"
    hybrid = "hybrid"


# ── Game requests ──────────────────────────────────────────────────────────────

class NewGameRequest(BaseModel):
    strategy_a: StrategyName = Field(..., description="Estrategia del jugador A (player 0)")
    strategy_b: StrategyName = Field(..., description="Estrategia del jugador B (player 1)")


# ── Per-turn metrics ───────────────────────────────────────────────────────────

class TurnMetricsModel(BaseModel):
    strategy: str
    turn: int
    time_ms: float
    nodes_expanded: int
    eval_calls: int
    max_depth: int
    move_chosen: str


# ── Real-time turn event (streamed via SSE) ────────────────────────────────────

class TurnEvent(BaseModel):
    type: str                          # "turn" | "game_over" | "error"
    turn: int
    player: int                        # 0 = A, 1 = B
    strategy: str
    move: str                          # e.g. "[3|5] → right" or "pass"
    drew_from_pool: bool
    board_length: int
    hand_size_a: int
    hand_size_b: int
    pool_size: int
    left_end: Optional[int]
    right_end: Optional[int]
    board_str: str
    metrics: Optional[TurnMetricsModel]
    is_terminal: bool


class GameOverEvent(BaseModel):
    type: str = "game_over"
    winner: Optional[int]              # 0 = A, 1 = B, -1 = draw, None = ongoing
    winner_name: str
    total_turns: int
    pip_sum_a: int
    pip_sum_b: int
    summary_a: dict
    summary_b: dict


# ── Session info ───────────────────────────────────────────────────────────────

class SessionInfo(BaseModel):
    session_id: str
    strategy_a: str
    strategy_b: str
    status: str                        # "active" | "finished"
    turn: int
    winner: Optional[int]


class SessionListResponse(BaseModel):
    sessions: list[SessionInfo]


# ── Benchmark ──────────────────────────────────────────────────────────────────

class BenchmarkRequest(BaseModel):
    n_games: int = Field(default=20, ge=1, le=200, description="Partidas por matchup")
    matchups: Optional[list[dict]] = Field(
        default=None,
        description="Lista de matchups. Si es null se usan los 5 estándar.",
    )


class MatchupResult(BaseModel):
    tag: str
    label: str
    agent_a: str
    agent_b: str
    n_games: int
    wins_a: int
    wins_b: int
    draws: int
    win_rate_a: float
    win_rate_b: float
    avg_turns: float
    turns_per_game: list[int]
    score_advantage_per_game: list[int]
    metrics_a: dict
    metrics_b: dict


class BenchmarkResult(BaseModel):
    run_id: str
    status: str                         # "running" | "done" | "error"
    n_games: int
    matchups: list[MatchupResult]
    total_time_s: float


# ── Metrics summary ────────────────────────────────────────────────────────────

class StrategySummary(BaseModel):
    strategy: str
    description: str
    avg_time_ms: float
    avg_nodes: float
    avg_evals: float
    avg_depth: float
    total_turns: int
