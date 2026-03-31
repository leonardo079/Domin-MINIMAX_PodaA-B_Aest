"""
strategies/base.py — Clases base para las estrategias de IA.
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

from app.core.game_state import GameState, Tile
from app.core.profiler import CostProfiler
from app.core.evaluator import evaluate

MINIMAX_DEFAULT_DEPTH = 7


class AgentStrategy(ABC):
    def __init__(self, player: int = 0):
        self.player = player
        self.profiler: Optional[CostProfiler] = None
        self.tree_recorder = None

    def set_profiler(self, profiler: CostProfiler) -> None:
        self.profiler = profiler

    def set_tree_recorder(self, recorder) -> None:
        self.tree_recorder = recorder

    @abstractmethod
    def decide(self, state: GameState) -> Optional[Tuple[Tile, str]]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class MinimaxStrategy(AgentStrategy):
    """
    Minimax con poda alpha-beta reutilizable.
    """

    use_manhattan: bool = True
    use_euclidean: bool = True
    use_pool: bool = False
    depth: int = MINIMAX_DEFAULT_DEPTH

    def decide(self, state: GameState) -> Optional[Tuple[Tile, str]]:
        if self.profiler:
            self.profiler.start_turn()

        rec = self.tree_recorder
        if rec:
            rec.reset()

        root_player = self.player
        hand = state.agent_hand if root_player == 0 else state.opponent_hand
        moves = state.valid_moves(hand)

        if not moves:
            if self.profiler:
                self.profiler.end_turn("pass")
            return None

        root_id = -1
        if rec:
            root_id = rec.add_node(
                None, 0, "ROOT",
                f"Turno jugador {self.player} ({self.name})",
                None, None,
            )

        ordered_moves = self._order_moves(
            state=state,
            moves=moves,
            player=root_player,
            maximizing=True,
        )

        best_score = float("-inf")
        best_move = ordered_moves[0]

        for tile, side in ordered_moves:
            if self.profiler:
                self.profiler.count_node()

            ns = state.apply_move(tile, side, root_player)
            score = self._minimax(
                state=ns,
                depth=self.depth - 1,
                alpha=float("-inf"),
                beta=float("inf"),
                current_player=1 - root_player,
                maximizing=False,
                _root_depth=self.depth,
                _parent_id=root_id,
                _move_label=f"{tile}→{side}",
            )

            if score > best_score:
                best_score = score
                best_move = (tile, side)

        if self.profiler:
            self.profiler.end_turn(str(best_move))

        return best_move

    def _minimax(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        current_player: int,
        maximizing: bool,
        _root_depth: int,
        _parent_id: int = -1,
        _move_label: str = "?",
    ) -> float:
        rec = self.tree_recorder
        node_id = -1
        ply = _root_depth - depth

        if rec:
            node_type = "MAX" if maximizing else "MIN"
            node_id = rec.add_node(
                _parent_id if _parent_id >= 0 else None,
                ply,
                node_type,
                _move_label,
                alpha if alpha != float("-inf") else None,
                beta if beta != float("inf") else None,
            )

        if self.profiler:
            self.profiler.count_node()
            self.profiler.update_depth(ply)

        if depth == 0 or state.is_terminal():
            if self.profiler:
                self.profiler.count_eval()

            val = evaluate(
                state,
                self.player,
                use_manhattan=self.use_manhattan,
                use_euclidean=self.use_euclidean,
                use_pool=self.use_pool,
            )

            if rec and node_id >= 0:
                rec.update_value(node_id, val)
            return val

        hand = state.agent_hand if current_player == 0 else state.opponent_hand
        moves = state.valid_moves(hand)

        if not moves:
            ns = state.apply_pass(current_player)
            val = self._minimax(
                state=ns,
                depth=depth - 1,
                alpha=alpha,
                beta=beta,
                current_player=1 - current_player,
                maximizing=not maximizing,
                _root_depth=_root_depth,
                _parent_id=node_id,
                _move_label="pass",
            )
            if rec and node_id >= 0:
                rec.update_value(node_id, val)
            return val

        moves = self._order_moves(
            state=state,
            moves=moves,
            player=current_player,
            maximizing=maximizing,
        )

        if maximizing:
            value = float("-inf")
            for i, (tile, side) in enumerate(moves):
                ns = state.apply_move(tile, side, current_player)
                child_val = self._minimax(
                    state=ns,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    current_player=1 - current_player,
                    maximizing=False,
                    _root_depth=_root_depth,
                    _parent_id=node_id,
                    _move_label=f"{tile}→{side}",
                )
                value = max(value, child_val)
                alpha = max(alpha, value)

                if beta <= alpha:
                    remaining = len(moves) - i - 1
                    if rec and node_id >= 0 and remaining > 0:
                        rec.add_node(
                            node_id, ply + 1, "PRUNED",
                            f"✂ {remaining} podado(s)",
                            alpha, beta, pruned=True,
                        )
                    break

            if rec and node_id >= 0:
                rec.update_value(node_id, value)
            return value

        value = float("inf")
        for i, (tile, side) in enumerate(moves):
            ns = state.apply_move(tile, side, current_player)
            child_val = self._minimax(
                state=ns,
                depth=depth - 1,
                alpha=alpha,
                beta=beta,
                current_player=1 - current_player,
                maximizing=True,
                _root_depth=_root_depth,
                _parent_id=node_id,
                _move_label=f"{tile}→{side}",
            )
            value = min(value, child_val)
            beta = min(beta, value)

            if beta <= alpha:
                remaining = len(moves) - i - 1
                if rec and node_id >= 0 and remaining > 0:
                    rec.add_node(
                        node_id, ply + 1, "PRUNED",
                        f"✂ {remaining} podado(s)",
                        alpha, beta, pruned=True,
                    )
                break

        if rec and node_id >= 0:
            rec.update_value(node_id, value)
        return value

    def _order_moves(
        self,
        state: GameState,
        moves: List[Tuple[Tile, str]],
        player: int,
        maximizing: bool,
    ) -> List[Tuple[Tile, str]]:
        """
        Ordenamiento heurístico para mejorar poda alpha-beta.
        MAX: primero mejores estados para el agente.
        MIN: primero peores estados para el agente.
        """
        scored = []
        for tile, side in moves:
            ns = state.apply_move(tile, side, player)
            score = evaluate(
                ns,
                self.player,
                use_manhattan=self.use_manhattan,
                use_euclidean=self.use_euclidean,
                use_pool=self.use_pool,
            )
            scored.append((score, (tile, side)))

        scored.sort(key=lambda x: x[0], reverse=maximizing)
        return [move for _, move in scored]