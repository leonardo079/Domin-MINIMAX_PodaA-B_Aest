"""
game_manager.py — Gestión de sesiones de juego en memoria.

Una sesión encapsula el estado completo de una partida:
  - GameState actual
  - Ambas estrategias con sus profilers
  - Historial de eventos por turno (para streaming y consulta posterior)
"""
import uuid
from typing import Optional
from app.core.game_state import GameState
from app.core.profiler import CostProfiler
from app.strategies import STRATEGIES, STRATEGY_DESCRIPTIONS


class GameSession:
    def __init__(self, session_id: str, strategy_a: str, strategy_b: str):
        self.session_id = session_id
        self.strategy_a_name = strategy_a
        self.strategy_b_name = strategy_b

        self.state = GameState.new_game()
        self.prof_a = CostProfiler(strategy_a)
        self.prof_b = CostProfiler(strategy_b)
        self.sa = STRATEGIES[strategy_a](player=0)
        self.sb = STRATEGIES[strategy_b](player=1)
        self.sa.set_profiler(self.prof_a)
        self.sb.set_profiler(self.prof_b)

        self.turn: int = 0
        self.status: str = "active"     # "active" | "finished"
        self.winner_id: Optional[int] = None
        self.turn_history: list[dict] = []

    # ── Ejecutar un turno ──────────────────────────────────────────────────

    def step(self) -> dict:
        """
        Ejecuta exactamente un turno y retorna el evento JSON correspondiente.
        Si la partida ya terminó retorna un evento 'game_over'.
        """
        if self.state.is_terminal() or self.status == "finished":
            return self._build_game_over_event()

        self.turn += 1
        cur = self.state.current_player
        strat = self.sa if cur == 0 else self.sb
        prof = self.prof_a if cur == 0 else self.prof_b
        hand = self.state.agent_hand if cur == 0 else self.state.opponent_hand

        # Robar del pozo si no hay jugadas
        moves = self.state.valid_moves(hand)
        drew = False
        if not moves and self.state.pool:
            self.state, moves = self.state.apply_draw_and_play(cur)
            drew = True

        # Decidir jugada
        result = strat.decide(self.state)

        if result is None:
            move_str = "pass"
            self.state = self.state.apply_pass(cur)
        else:
            tile, side = result
            move_str = f"{tile} → {side}"
            self.state = self.state.apply_move(tile, side, cur)

        # Capturar métricas del turno
        last_metric = prof.last_metric_dict()

        event = {
            "type": "turn",
            "turn": self.turn,
            "player": cur,
            "strategy": strat.name,
            "move": move_str,
            "drew_from_pool": drew,
            "board_length": len(self.state.board),
            "hand_size_a": len(self.state.agent_hand),
            "hand_size_b": len(self.state.opponent_hand),
            "pool_size": self.state.pool_size(),
            "left_end": self.state.left_end,
            "right_end": self.state.right_end,
            "board_str": self.state.board_str(),
            "metrics": last_metric,
            "is_terminal": self.state.is_terminal(),
        }
        self.turn_history.append(event)

        if self.state.is_terminal():
            self.status = "finished"
            self.winner_id = self.state.winner()

        return event

    def _build_game_over_event(self) -> dict:
        winner_id = self.state.winner()
        self.winner_id = winner_id
        self.status = "finished"

        winner_map = {0: self.strategy_a_name, 1: self.strategy_b_name, -1: "draw", None: "unknown"}
        return {
            "type": "game_over",
            "winner": winner_id,
            "winner_name": winner_map.get(winner_id, "unknown"),
            "total_turns": self.turn,
            "pip_sum_a": self.state.pip_sum(0),
            "pip_sum_b": self.state.pip_sum(1),
            "summary_a": self.prof_a.summary(),
            "summary_b": self.prof_b.summary(),
        }

    def get_state_snapshot(self) -> dict:
        return {
            "session_id": self.session_id,
            "strategy_a": self.strategy_a_name,
            "strategy_b": self.strategy_b_name,
            "status": self.status,
            "turn": self.turn,
            "winner": self.winner_id,
            **self.state.to_dict(),
        }

    def get_metrics_history(self) -> dict:
        return {
            "session_id": self.session_id,
            "strategy_a": self.strategy_a_name,
            "strategy_b": self.strategy_b_name,
            "metrics_a": self.prof_a.all_metrics_list(),
            "metrics_b": self.prof_b.all_metrics_list(),
            "summary_a": self.prof_a.summary(),
            "summary_b": self.prof_b.summary(),
        }

    def to_info(self) -> dict:
        return {
            "session_id": self.session_id,
            "strategy_a": self.strategy_a_name,
            "strategy_b": self.strategy_b_name,
            "status": self.status,
            "turn": self.turn,
            "winner": self.winner_id,
        }


# ── Registro global de sesiones ────────────────────────────────────────────────

_sessions: dict[str, GameSession] = {}


def create_session(strategy_a: str, strategy_b: str) -> GameSession:
    sid = str(uuid.uuid4())
    session = GameSession(sid, strategy_a, strategy_b)
    _sessions[sid] = session
    return session


def get_session(session_id: str) -> Optional[GameSession]:
    return _sessions.get(session_id)


def delete_session(session_id: str) -> bool:
    if session_id in _sessions:
        del _sessions[session_id]
        return True
    return False


def list_sessions() -> list[dict]:
    return [s.to_info() for s in _sessions.values()]
