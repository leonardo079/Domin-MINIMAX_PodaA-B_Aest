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
from app.core.tree_recorder import TreeRecorder
from app.strategies import STRATEGIES, STRATEGY_DESCRIPTIONS


class GameSession:
    def __init__(self, session_id: str, strategy_a: str, strategy_b: str,
                 game_mode: str = "agent_vs_agent"):
        self.session_id = session_id
        self.strategy_a_name = strategy_a
        self.strategy_b_name = strategy_b
        self.game_mode = game_mode

        self.state = GameState.new_game()
        self.prof_a = CostProfiler(strategy_a)
        self.prof_b = CostProfiler(strategy_b)
        self.sa = STRATEGIES[strategy_a](player=0)
        self.sb = STRATEGIES[strategy_b](player=1)
        self.sa.set_profiler(self.prof_a)
        self.sb.set_profiler(self.prof_b)

        # Árboles de búsqueda — uno por agente
        self.tree_rec_a = TreeRecorder()
        self.tree_rec_b = TreeRecorder()
        self.sa.set_tree_recorder(self.tree_rec_a)
        self.sb.set_tree_recorder(self.tree_rec_b)
        self.last_tree_a: Optional[dict] = None
        self.last_tree_b: Optional[dict] = None

        self.turn: int = 0
        self.status: str = "active"     # "active" | "finished"
        self.winner_id: Optional[int] = None
        self.turn_history: list[dict] = []

    # ── Ejecutar un turno (IA) ────────────────────────────────────────────────

    def step(self) -> dict:
        """
        Ejecuta exactamente un turno de IA y retorna el evento JSON.
        En modo agent_vs_human, si es el turno del humano lanza ValueError.
        Si la partida ya terminó retorna un evento 'game_over'.
        """
        if self.state.is_terminal() or self.status == "finished":
            return self._build_game_over_event()

        cur = self.state.current_player

        # En modo humano, el turno del jugador 1 lo maneja human_move()
        if self.game_mode == "agent_vs_human" and cur == 1:
            raise ValueError("Es el turno del humano — usa POST /human-move")

        self.turn += 1
        strat = self.sa if cur == 0 else self.sb
        prof = self.prof_a if cur == 0 else self.prof_b
        hand = self.state.agent_hand if cur == 0 else self.state.opponent_hand

        # Robar del pozo si no hay jugadas
        moves = self.state.valid_moves(hand)
        drew = False
        if not moves and self.state.pool:
            self.state, moves = self.state.apply_draw_and_play(cur)
            drew = True

        # Decidir jugada (el recorder interno ya fue reseteado en decide())
        result = strat.decide(self.state)

        # Capturar snapshot del árbol generado en este turno
        if cur == 0:
            self.last_tree_a = self.tree_rec_a.to_dict()
        else:
            self.last_tree_b = self.tree_rec_b.to_dict()

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
        elif self.game_mode == "agent_vs_human" and self.state.current_player == 1:
            self.status = "waiting_human"
        else:
            self.status = "active"

        return event

    # ── Turno del humano ───────────────────────────────────────────────────────

    def human_move(self, tile_a: int, tile_b: int, side: str) -> dict:
        """
        Aplica la jugada que el humano envía desde el frontend.
        Solo válido en modo agent_vs_human cuando current_player == 1.
        """
        if self.game_mode != "agent_vs_human":
            raise ValueError("Esta sesión no es de modo agent_vs_human")
        if self.state.current_player != 1:
            raise ValueError("No es el turno del humano")
        if self.status == "finished":
            raise ValueError("La partida ya terminó")

        self.turn += 1
        hand = self.state.opponent_hand

        # Robar del pozo si no hay jugadas
        moves = self.state.valid_moves(hand)
        drew = False
        if not moves and self.state.pool:
            self.state, moves = self.state.apply_draw_and_play(1)
            drew = True
            hand = self.state.opponent_hand

        # Buscar la ficha enviada en la mano del humano
        chosen_tile = next(
            (t for t in hand if
             (t.a == tile_a and t.b == tile_b) or (t.a == tile_b and t.b == tile_a)),
            None,
        )
        if chosen_tile is None:
            raise ValueError(f"La ficha [{tile_a}|{tile_b}] no está en tu mano")

        # Validar que la jugada sea legal
        valid = self.state.valid_moves(hand)
        if (chosen_tile, side) not in valid:
            valid_str = ", ".join(f"{t} → {s}" for t, s in valid) or "ninguna (pasa)"
            raise ValueError(f"Jugada inválida. Jugadas válidas: {valid_str}")

        move_str = f"{chosen_tile} → {side}"
        self.state = self.state.apply_move(chosen_tile, side, 1)

        event = {
            "type": "turn",
            "turn": self.turn,
            "player": 1,
            "strategy": "human",
            "move": move_str,
            "drew_from_pool": drew,
            "board_length": len(self.state.board),
            "hand_size_a": len(self.state.agent_hand),
            "hand_size_b": len(self.state.opponent_hand),
            "pool_size": self.state.pool_size(),
            "left_end": self.state.left_end,
            "right_end": self.state.right_end,
            "board_str": self.state.board_str(),
            "metrics": None,
            "is_terminal": self.state.is_terminal(),
        }
        self.turn_history.append(event)

        if self.state.is_terminal():
            self.status = "finished"
            self.winner_id = self.state.winner()
        else:
            self.status = "active"  # turno de la IA

        return event

    def human_pass(self) -> dict:
        """El humano pasa (sin jugadas ni pozo disponible)."""
        if self.game_mode != "agent_vs_human":
            raise ValueError("Esta sesión no es de modo agent_vs_human")
        if self.state.current_player != 1:
            raise ValueError("No es el turno del humano")

        hand = self.state.opponent_hand
        moves = self.state.valid_moves(hand)
        if moves or self.state.pool:
            raise ValueError("No puedes pasar si tienes jugadas disponibles o hay pozo")

        self.turn += 1
        self.state = self.state.apply_pass(1)

        event = {
            "type": "turn",
            "turn": self.turn,
            "player": 1,
            "strategy": "human",
            "move": "pass",
            "drew_from_pool": False,
            "board_length": len(self.state.board),
            "hand_size_a": len(self.state.agent_hand),
            "hand_size_b": len(self.state.opponent_hand),
            "pool_size": self.state.pool_size(),
            "left_end": self.state.left_end,
            "right_end": self.state.right_end,
            "board_str": self.state.board_str(),
            "metrics": None,
            "is_terminal": self.state.is_terminal(),
        }
        self.turn_history.append(event)

        if self.state.is_terminal():
            self.status = "finished"
            self.winner_id = self.state.winner()
        else:
            self.status = "active"

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
        snap = {
            "session_id": self.session_id,
            "strategy_a": self.strategy_a_name,
            "strategy_b": self.strategy_b_name,
            "game_mode": self.game_mode,
            "status": self.status,
            "turn": self.turn,
            "winner": self.winner_id,
            **self.state.to_dict(),
        }
        # En modo humano, exponer las fichas y jugadas válidas del humano
        if self.game_mode == "agent_vs_human" and self.status == "waiting_human":
            hand = self.state.opponent_hand
            valid = self.state.valid_moves(hand)
            snap["human_hand"] = [{"a": t.a, "b": t.b, "pips": t.pips()} for t in hand]
            snap["human_valid_moves"] = [
                {"tile": {"a": t.a, "b": t.b}, "side": s} for t, s in valid
            ]
        return snap

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

    def get_last_trees(self) -> dict:
        """Retorna los árboles de búsqueda del último turno de cada agente."""
        return {
            "session_id":  self.session_id,
            "strategy_a":  self.strategy_a_name,
            "strategy_b":  self.strategy_b_name,
            "game_mode":   self.game_mode,
            "turn":        self.turn,
            "tree_a":      self.last_tree_a,   # árbol del agente A (siempre IA)
            "tree_b":      self.last_tree_b,   # árbol del agente B (IA o None si human)
        }

    def to_info(self) -> dict:
        return {
            "session_id": self.session_id,
            "game_mode": self.game_mode,
            "strategy_a": self.strategy_a_name,
            "strategy_b": self.strategy_b_name,
            "status": self.status,
            "turn": self.turn,
            "winner": self.winner_id,
        }


# ── Registro global de sesiones ────────────────────────────────────────────────

_sessions: dict[str, GameSession] = {}


def create_session(strategy_a: str, strategy_b: Optional[str],
                   game_mode: str = "agent_vs_agent") -> GameSession:
    sid = str(uuid.uuid4())
    session = GameSession(sid, strategy_a, strategy_b, game_mode)
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
