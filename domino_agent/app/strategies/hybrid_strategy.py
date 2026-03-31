"""
hybrid_strategy.py — estrategia híbrida heurístico-adversarial.

Enfoque:
1. Se generan las jugadas válidas.
2. Se calcula una heurística compuesta para cada jugada.
3. Se seleccionan solo unas pocas candidatas de mayor calidad heurística.
4. Sobre esas candidatas se ejecuta una verificación Minimax corta con poda alpha-beta.
5. La decisión final combina principalmente la heurística y, en menor proporción,
   la validación adversarial local.

Este diseño es más apropiado para dominó con pozo que un Minimax profundo puro,
porque da mayor peso a señales prácticas bajo información imperfecta y usa Minimax
solo como corrector local, no como motor dominante.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from app.core.game_state import GameState, Tile
from app.strategies.base import AgentStrategy
from app.core.evaluator import (
    evaluate,
    euclidean_distance,
    manhattan_distance,
    opponent_blocking_score,
    pool_opportunity_score,
)

# Configuración principal del híbrido
CANDIDATE_COUNT = 4
MINIMAX_VERIFY_DEPTH = 2
HEURISTIC_WEIGHT = 0.72
MINIMAX_WEIGHT = 0.28


class HybridStrategy(AgentStrategy):

    @property
    def name(self) -> str:
        return "hybrid"

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
                None,
                0,
                "ROOT",
                f"Turno jugador {self.player} (Hybrid)",
                None,
                None,
            )

        ranked = self._rank_moves(state, moves, root_player, root_id)
        candidates = ranked[: min(CANDIDATE_COUNT, len(ranked))]

        best_move = candidates[0][0]
        best_final_score = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        verify_parent = -1
        if rec:
            verify_parent = rec.add_node(
                root_id if root_id >= 0 else None,
                1,
                "VERIFY",
                f"Verificación Minimax corta sobre {len(candidates)} candidata(s)",
                None,
                None,
            )

        for rank_idx, ((tile, side), next_state, heuristic_score, heuristic_node_id) in enumerate(candidates, start=1):
            if self.profiler:
                self.profiler.count_node()

            minimax_score = self._minimax(
                state=next_state,
                depth=MINIMAX_VERIFY_DEPTH,
                alpha=alpha,
                beta=beta,
                current_player=1 - root_player,
                maximizing=False,
                _root_depth=MINIMAX_VERIFY_DEPTH + 1,
                _parent_id=verify_parent if verify_parent >= 0 else heuristic_node_id,
                _move_label=f"#{rank_idx} {tile}→{side}",
            )

            final_score = HEURISTIC_WEIGHT * heuristic_score + MINIMAX_WEIGHT * minimax_score

            if rec:
                combo_id = rec.add_node(
                    verify_parent if verify_parent >= 0 else root_id,
                    2,
                    "COMBINE",
                    f"{tile}→{side}",
                    round(heuristic_score, 4),
                    round(minimax_score, 4),
                )
                rec.update_value(combo_id, round(final_score, 4))

            if final_score > best_final_score:
                best_final_score = final_score
                best_move = (tile, side)

            alpha = max(alpha, best_final_score)

        if self.profiler:
            self.profiler.end_turn(str(best_move))
        return best_move

    def _rank_moves(
        self,
        state: GameState,
        moves: List[Tuple[Tile, str]],
        player: int,
        root_id: int,
    ) -> List[Tuple[Tuple[Tile, str], GameState, float, int]]:
        rec = self.tree_recorder
        ranked: List[Tuple[Tuple[Tile, str], GameState, float, int]] = []

        rank_parent = -1
        if rec:
            rank_parent = rec.add_node(
                root_id if root_id >= 0 else None,
                1,
                "HEURISTIC_PHASE",
                f"Ranking heurístico ({len(moves)} jugadas)",
                None,
                None,
            )

        for tile, side in moves:
            next_state = state.apply_move(tile, side, player)
            score = self._heuristic_move_score(state, next_state, tile)

            node_id = rank_parent
            if rec:
                node_id = rec.add_node(
                    rank_parent if rank_parent >= 0 else None,
                    2,
                    "HEURISTIC",
                    f"{tile}→{side}",
                    None,
                    None,
                )
                rec.update_value(node_id, round(score, 4))

            ranked.append(((tile, side), next_state, score, node_id))

        ranked.sort(key=lambda item: item[2], reverse=True)
        return ranked

    def _heuristic_move_score(
        self,
        current_state: GameState,
        next_state: GameState,
        tile: Tile,
    ) -> float:
        """
        Puntaje principal del híbrido.
        Mayor es mejor.
        La heurística pesa más que la verificación Minimax.
        """
        value = evaluate(
            next_state,
            self.player,
            use_manhattan=True,
            use_euclidean=True,
            use_pool=True,
        )
        block = opponent_blocking_score(next_state, self.player)
        pool = pool_opportunity_score(next_state, self.player) if next_state.pool_size() > 0 else 0.0
        my_hand = next_state.agent_hand if self.player == 0 else next_state.opponent_hand
        my_moves_after = len(next_state.valid_moves(my_hand))

        m_dist = manhattan_distance(next_state, self.player)
        e_dist = euclidean_distance(next_state, self.player)

        # Incentivo por descargar fichas pesadas o dobles, útil en finales/bloqueos.
        pip_value = self._tile_pip_value(tile)
        t0, t1 = self._tile_values(tile)
        double_bonus = 1.0 if t0 == t1 else 0.0

        # Penalización suave si tras la jugada quedamos con poca movilidad.
        mobility_penalty = 0.35 if my_moves_after <= 1 else 0.0

        # Componente base normalizado con pesos prácticos para dominó con incertidumbre.
        return (
            0.46 * value
            + 0.22 * block
            + 0.10 * pool
            + 0.10 * min(my_moves_after, 6) / 6.0
            + 0.07 * min(pip_value, 12) / 12.0
            + 0.05 * double_bonus
            - 0.08 * min(m_dist, 12) / 12.0
            - 0.04 * min(e_dist, 12.0) / 12.0
            - mobility_penalty
        )

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
            value = evaluate(
                state,
                self.player,
                use_manhattan=True,
                use_euclidean=True,
                use_pool=True,
            )
            if rec and node_id >= 0:
                rec.update_value(node_id, value)
            return value

        hand = state.agent_hand if current_player == 0 else state.opponent_hand
        moves = state.valid_moves(hand)

        if not moves:
            next_state = state.apply_pass(current_player)
            value = self._minimax(
                state=next_state,
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
                rec.update_value(node_id, value)
            return value

        ordered_moves = self._order_moves_for_verify(state, moves, current_player, maximizing)

        if maximizing:
            value = float("-inf")
            for i, (tile, side) in enumerate(ordered_moves):
                next_state = state.apply_move(tile, side, current_player)
                child_val = self._minimax(
                    state=next_state,
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
                    if rec and node_id >= 0 and i < len(ordered_moves) - 1:
                        rec.add_node(
                            node_id,
                            ply + 1,
                            "PRUNED",
                            f"✂ {len(ordered_moves) - i - 1} podado(s)",
                            alpha,
                            beta,
                            pruned=True,
                        )
                    break
            if rec and node_id >= 0:
                rec.update_value(node_id, value)
            return value

        value = float("inf")
        for i, (tile, side) in enumerate(ordered_moves):
            next_state = state.apply_move(tile, side, current_player)
            child_val = self._minimax(
                state=next_state,
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
                if rec and node_id >= 0 and i < len(ordered_moves) - 1:
                    rec.add_node(
                        node_id,
                        ply + 1,
                        "PRUNED",
                        f"✂ {len(ordered_moves) - i - 1} podado(s)",
                        alpha,
                        beta,
                        pruned=True,
                    )
                break

        if rec and node_id >= 0:
            rec.update_value(node_id, value)
        return value

    def _order_moves_for_verify(
        self,
        state: GameState,
        moves: List[Tuple[Tile, str]],
        player: int,
        maximizing: bool,
    ) -> List[Tuple[Tile, str]]:
        scored = []
        for tile, side in moves:
            next_state = state.apply_move(tile, side, player)
            score = evaluate(
                next_state,
                self.player,
                use_manhattan=True,
                use_euclidean=True,
                use_pool=True,
            )
            scored.append((score, (tile, side)))

        scored.sort(key=lambda item: item[0], reverse=maximizing)
        return [move for _, move in scored]

    @staticmethod
    def _tile_values(tile: Tile) -> Tuple[int, int]:
        """
        Soporta distintos formatos de ficha:
        - objetos con atributos a/b
        - objetos con atributos left/right o left_val/right_val
        - tuplas/listas indexables
        """
        for left_name, right_name in (("a", "b"), ("left", "right"), ("left_val", "right_val")):
            if hasattr(tile, left_name) and hasattr(tile, right_name):
                try:
                    return int(getattr(tile, left_name)), int(getattr(tile, right_name))
                except Exception:
                    pass

        try:
            return int(tile[0]), int(tile[1])
        except Exception:
            return 0, 0

    @classmethod
    def _tile_pip_value(cls, tile: Tile) -> int:
        left, right = cls._tile_values(tile)
        return left + right
