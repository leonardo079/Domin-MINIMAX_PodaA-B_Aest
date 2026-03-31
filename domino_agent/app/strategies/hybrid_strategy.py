import math
from typing import Optional, Tuple
from app.core.game_state import GameState, Tile
from app.strategies.base import AgentStrategy
from app.core.evaluator import (
    evaluate,
    euclidean_distance,
    manhattan_distance,
    opponent_blocking_score,
    pool_opportunity_score,
)

MIN_DEPTH = 3
BASE_DEPTH = 4
MAX_DEPTH = 5
TOP_K_MIN = 2
TOP_K_MAX = 5


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

        hand = state.agent_hand if self.player == 0 else state.opponent_hand
        moves = state.valid_moves(hand)
        if not moves:
            if self.profiler:
                self.profiler.end_turn("pass")
            return None

        # ── Nodo raíz ────────────────────────────────────────────────────────
        root_id = -1
        if rec:
            root_id = rec.add_node(None, 0, "ROOT",
                                   f"Turno jugador {self.player} (Híbrido)",
                                   None, None)

        # Fase 1: A* ordering sobre jugadas candidatas.
        search_depth = self._select_depth(state)
        top_k = self._select_top_k(len(moves), search_depth)
        g = 7 - len(hand)
        astar_phase_id = -1
        if rec:
            astar_phase_id = rec.add_node(root_id if root_id >= 0 else None,
                                          1, "ASTAR_PHASE",
                                          f"Ranking A* ({len(moves)} candidatas, d={search_depth})",
                                          None, None)

        scored = []
        for tile, side in moves:
            ns = state.apply_move(tile, side, self.player)
            h = self._heuristic(ns)
            f_val = g + h
            scored.append((f_val, (tile, side), ns))
            if self.profiler:
                self.profiler.count_node()
            if rec:
                rec.add_node(astar_phase_id if astar_phase_id >= 0 else None,
                             2, "ASTAR_RANK",
                             f"{tile}\u2192{side}  f={f_val:.2f}",
                             alpha=f_val, beta=round(h, 4), value=round(g, 4))

        scored.sort(key=lambda x: x[0])
        top_moves = [(m, ns) for _, m, ns in scored[:top_k]]

        # Fase 2: Minimax con poda alpha-beta sobre top-k.
        mm_phase_id = -1
        if rec:
            mm_phase_id = rec.add_node(root_id if root_id >= 0 else None,
                                       1, "MINIMAX_PHASE",
                                       f"Minimax alpha-beta (top {top_k} candidatas, d={search_depth})",
                                       None, None)

        best_score = float('-inf')
        best_move = top_moves[0][0]

        for (tile, side), ns in top_moves:
            if self.profiler:
                self.profiler.count_node()
            score = self._minimax(ns, search_depth - 1, float('-inf'), float('inf'), False,
                                  _root_depth=search_depth,
                                  _parent_id=mm_phase_id,
                                  _move_label=f"{tile}\u2192{side}")
            if score > best_score:
                best_score = score
                best_move = (tile, side)

        if self.profiler:
            self.profiler.end_turn(str(best_move))
        return best_move

    def _minimax(self, state, depth, alpha, beta, maximizing,
                 _root_depth: int,
                 _parent_id: int = -1, _move_label: str = "?"):
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
                alpha if alpha != float('-inf') else None,
                beta  if beta  != float('inf')  else None,
            )

        if self.profiler:
            self.profiler.count_node()
            self.profiler.update_depth(ply)

        if depth == 0 or state.is_terminal():
            if self.profiler:
                self.profiler.count_eval()
            val = evaluate(state, self.player,
                           use_manhattan=True, use_euclidean=True)
            if rec and node_id >= 0:
                rec.update_value(node_id, val)
            return val

        hand = state.agent_hand if maximizing else state.opponent_hand
        player = self.player if maximizing else 1 - self.player
        moves = state.valid_moves(hand)

        # A* ordering dentro del minimax
        if moves:
            g_local = 7 - len(hand)
            scored = []
            for tile, side in moves:
                ns_temp = state.apply_move(tile, side, player)
                h = self._heuristic(ns_temp)
                scored.append((g_local + h, tile, side))
            scored.sort(key=lambda x: x[0])
            moves = [(t, s) for _, t, s in scored]

        if not moves:
            ns = state.apply_pass(player)
            val = self._minimax(ns, depth - 1, alpha, beta, not maximizing,
                                _root_depth=_root_depth,
                                _parent_id=node_id, _move_label="pass")
            if rec and node_id >= 0:
                rec.update_value(node_id, val)
            return val

        if maximizing:
            val = float('-inf')
            for i, (tile, side) in enumerate(moves):
                ns = state.apply_move(tile, side, player)
                val = max(val, self._minimax(ns, depth - 1, alpha, beta, False,
                                             _root_depth=_root_depth,
                                             _parent_id=node_id,
                                             _move_label=f"{tile}\u2192{side}"))
                alpha = max(alpha, val)
                if beta <= alpha:
                    remaining = len(moves) - i - 1
                    if rec and node_id >= 0 and remaining > 0:
                        rec.add_node(node_id, ply + 1, "PRUNED",
                                     f"\u2702 {remaining} podado(s)",
                                     alpha, beta, pruned=True)
                    break
            if rec and node_id >= 0:
                rec.update_value(node_id, val)
            return val
        else:
            val = float('inf')
            for i, (tile, side) in enumerate(moves):
                ns = state.apply_move(tile, side, player)
                val = min(val, self._minimax(ns, depth - 1, alpha, beta, True,
                                             _root_depth=_root_depth,
                                             _parent_id=node_id,
                                             _move_label=f"{tile}\u2192{side}"))
                beta = min(beta, val)
                if beta <= alpha:
                    remaining = len(moves) - i - 1
                    if rec and node_id >= 0 and remaining > 0:
                        rec.add_node(node_id, ply + 1, "PRUNED",
                                     f"\u2702 {remaining} podado(s)",
                                     alpha, beta, pruned=True)
                    break
            if rec and node_id >= 0:
                rec.update_value(node_id, val)
            return val

    def _heuristic(self, state) -> float:
        m = manhattan_distance(state, self.player)
        e = euclidean_distance(state, self.player)
        pool = pool_opportunity_score(state, self.player)
        block = opponent_blocking_score(state, self.player)
        # Menor es mejor para A*: distancia pesa en contra, bloqueo/pozo favorecen.
        return 0.45 * m + 0.45 * e - 0.05 * block - 0.05 * pool

    def _select_depth(self, state: GameState) -> int:
        hand = state.agent_hand if self.player == 0 else state.opponent_hand
        opp_hand = state.opponent_hand if self.player == 0 else state.agent_hand

        my_mobility = len(state.valid_moves(hand))
        opp_mobility = len(state.valid_moves(opp_hand))
        pressure = max(0, opp_mobility - my_mobility)

        depth = BASE_DEPTH
        if len(hand) <= 3 or len(opp_hand) <= 3:
            depth += 1
        if state.pool_size() > 0 and pressure >= 2:
            depth += 1
        if my_mobility >= 8:
            depth -= 1
        if state.pool_size() == 0 and my_mobility <= 2:
            depth -= 1

        return max(MIN_DEPTH, min(MAX_DEPTH, depth))

    def _select_top_k(self, n_moves: int, search_depth: int) -> int:
        ratio = 0.65 if search_depth <= BASE_DEPTH else 0.5
        k = math.ceil(n_moves * ratio)
        return max(TOP_K_MIN, min(TOP_K_MAX, k))
