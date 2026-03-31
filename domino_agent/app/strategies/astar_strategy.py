"""
astar_strategy.py — búsqueda A* sobre el espacio de estados.

Nota teórica importante:
A* es un algoritmo de búsqueda de caminos de un solo agente. En un juego
adversarial, A* no modela por sí mismo la intención de un rival; para eso la
teoría correcta es Minimax. Aquí se implementa A* "de verdad" a nivel
mecánico: frontera abierta, conjunto cerrado, costo acumulado g(n), heurística
h(n) y evaluación f(n)=g(n)+h(n).

En este dominio, el rival se trata como parte del espacio de estados y no como
un agente minimizador. Por eso esta estrategia sirve como búsqueda informada,
pero no sustituye a Minimax en teoría de juegos.
"""
from __future__ import annotations

import heapq
import itertools
from typing import Dict, List, Optional, Tuple

from app.core.game_state import GameState, Tile
from app.strategies.base import AgentStrategy
from app.core.evaluator import evaluate, euclidean_distance, manhattan_distance

MAX_EXPANSIONS = 250
MAX_SEARCH_DEPTH = 8


class AStarStrategy(AgentStrategy):

    @property
    def name(self) -> str:
        return "astar"

    def decide(self, state: GameState) -> Optional[Tuple[Tile, str]]:
        if self.profiler:
            self.profiler.start_turn()

        rec = self.tree_recorder
        if rec:
            rec.reset()

        my_hand = state.agent_hand if self.player == 0 else state.opponent_hand
        moves = state.valid_moves(my_hand)
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
                f"Turno jugador {self.player} (A*)",
                None,
                None,
            )

        counter = itertools.count()
        open_heap: List[Tuple[float, float, int, int, GameState, Optional[Tuple[Tile, str]], int]] = []
        best_g: Dict[Tuple, float] = {}

        root_key = self._state_key(state)
        best_g[root_key] = 0.0
        root_h = self._heuristic_cost(state)
        heapq.heappush(open_heap, (root_h, root_h, next(counter), root_id, state, None, 0))

        best_first_move: Optional[Tuple[Tile, str]] = None
        best_fallback_f = float("inf")
        expansions = 0

        while open_heap and expansions < MAX_EXPANSIONS:
            f_val, _, _, parent_id, current_state, first_move, depth = heapq.heappop(open_heap)
            state_key = self._state_key(current_state)
            current_g = best_g.get(state_key, float("inf"))

            if self.profiler:
                self.profiler.count_node()
                self.profiler.update_depth(depth)

            if current_state.is_terminal():
                if self.profiler:
                    self.profiler.count_eval()
                if current_state.winner() == self.player:
                    if self.profiler:
                        self.profiler.end_turn(str(first_move))
                    return first_move
                continue

            if depth >= MAX_SEARCH_DEPTH:
                continue

            current_player = current_state.current_player
            hand = current_state.agent_hand if current_player == 0 else current_state.opponent_hand
            valid_moves = current_state.valid_moves(hand)

            if not valid_moves:
                next_state = current_state.apply_pass(current_player)
                next_key = self._state_key(next_state)
                step_cost = self._step_cost(current_state, next_state)
                tentative_g = current_g + step_cost

                if tentative_g < best_g.get(next_key, float("inf")):
                    best_g[next_key] = tentative_g
                    h_val = self._heuristic_cost(next_state)
                    next_first = first_move
                    next_f = tentative_g + h_val
                    heapq.heappush(
                        open_heap,
                        (next_f, h_val, next(counter), parent_id, next_state, next_first, depth + 1),
                    )
                continue

            ordered_moves = self._order_moves(current_state, valid_moves, current_player)
            expansions += 1

            for tile, side in ordered_moves:
                next_state = current_state.apply_move(tile, side, current_player)
                next_key = self._state_key(next_state)
                step_cost = self._step_cost(current_state, next_state)
                tentative_g = current_g + step_cost

                if tentative_g >= best_g.get(next_key, float("inf")):
                    continue

                best_g[next_key] = tentative_g
                h_val = self._heuristic_cost(next_state)
                next_f = tentative_g + h_val
                next_first = first_move if first_move is not None else (tile, side)

                node_id = parent_id
                if rec:
                    node_id = rec.add_node(
                        parent_id if parent_id >= 0 else None,
                        depth + 1,
                        "A*",
                        f"{tile}→{side}",
                        round(tentative_g, 4),
                        round(h_val, 4),
                    )
                    rec.update_value(node_id, round(next_f, 4))

                if current_player == self.player and next_f < best_fallback_f:
                    best_fallback_f = next_f
                    best_first_move = next_first

                heapq.heappush(
                    open_heap,
                    (next_f, h_val, next(counter), node_id, next_state, next_first, depth + 1),
                )

        if best_first_move is None:
            best_first_move = self._best_immediate_move(state, moves)

        if self.profiler:
            self.profiler.end_turn(str(best_first_move))
        return best_first_move

    def _best_immediate_move(
        self,
        state: GameState,
        moves: List[Tuple[Tile, str]],
    ) -> Tuple[Tile, str]:
        best_move = moves[0]
        best_f = float("inf")
        for tile, side in self._order_moves(state, moves, self.player):
            next_state = state.apply_move(tile, side, self.player)
            f_val = 1.0 + self._heuristic_cost(next_state)
            if f_val < best_f:
                best_f = f_val
                best_move = (tile, side)
        return best_move

    def _step_cost(self, current_state: GameState, next_state: GameState) -> float:
        current_hand = current_state.agent_hand if self.player == 0 else current_state.opponent_hand
        next_hand = next_state.agent_hand if self.player == 0 else next_state.opponent_hand
        hand_delta = len(current_hand) - len(next_hand)

        if hand_delta > 0:
            return 1.0
        if next_state.current_player == self.player:
            return 0.5
        return 0.8

    def _heuristic_cost(self, state: GameState) -> float:
        if state.is_terminal():
            winner = state.winner()
            if winner == self.player:
                return 0.0
            if winner == -1:
                return 0.5
            return 1000.0

        value = evaluate(
            state,
            self.player,
            use_manhattan=True,
            use_euclidean=True,
            use_pool=True,
        )
        my_hand = state.agent_hand if self.player == 0 else state.opponent_hand
        hand_penalty = len(my_hand) / 7.0
        dist_penalty = 0.08 * manhattan_distance(state, self.player)
        dist_penalty += 0.04 * euclidean_distance(state, self.player)

        normalized_value_cost = (1.0 - value) / 2.0
        return max(0.0, normalized_value_cost + hand_penalty + dist_penalty)

    def _order_moves(
        self,
        state: GameState,
        moves: List[Tuple[Tile, str]],
        player: int,
    ) -> List[Tuple[Tile, str]]:
        scored = []
        for tile, side in moves:
            next_state = state.apply_move(tile, side, player)
            score = self._heuristic_cost(next_state)
            scored.append((score, (tile, side)))
        scored.sort(key=lambda item: item[0])
        return [move for _, move in scored]

    def _state_key(self, state: GameState) -> Tuple:
        board = tuple((ot.left_val, ot.right_val) for ot in getattr(state, "board", []))
        agent_hand = tuple(sorted((t.a, t.b) for t in state.agent_hand))
        opponent_hand = tuple(sorted((t.a, t.b) for t in state.opponent_hand))
        pool = tuple((t.a, t.b) for t in getattr(state, "pool", []))
        unknown = tuple(sorted((t.a, t.b) for t in getattr(state, "unknown_tiles", [])))
        return (
            board,
            state.left_end,
            state.right_end,
            agent_hand,
            opponent_hand,
            pool,
            unknown,
            state.current_player,
            state.pass_count,
        )
