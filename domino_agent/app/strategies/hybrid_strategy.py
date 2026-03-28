from typing import Optional, Tuple
from app.core.game_state import GameState, Tile
from app.strategies.base import AgentStrategy
from app.core.evaluator import evaluate, manhattan_distance, euclidean_distance

DEPTH = 4
TOP_K = 4


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

        # ── Fase 1: A* — ordenar jugadas por f(n) = g + h ───────────────────
        g = 7 - len(hand)
        astar_phase_id = -1
        if rec:
            astar_phase_id = rec.add_node(root_id if root_id >= 0 else None,
                                          1, "ASTAR_PHASE",
                                          f"Ranking A* ({len(moves)} candidatas)",
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
        top_moves = [(m, ns) for _, m, ns in scored[:TOP_K]]

        # ── Fase 2: Minimax + α-β sobre las top-K candidatas ─────────────────
        mm_phase_id = -1
        if rec:
            mm_phase_id = rec.add_node(root_id if root_id >= 0 else None,
                                       1, "MINIMAX_PHASE",
                                       f"Minimax α-β (top {TOP_K} candidatas)",
                                       None, None)

        best_score = float('-inf')
        best_move = top_moves[0][0]

        for (tile, side), ns in top_moves:
            if self.profiler:
                self.profiler.count_node()
            score = self._minimax(ns, DEPTH - 1, float('-inf'), float('inf'), False,
                                  _parent_id=mm_phase_id,
                                  _move_label=f"{tile}\u2192{side}")
            if score > best_score:
                best_score = score
                best_move = (tile, side)

        if self.profiler:
            self.profiler.end_turn(str(best_move))
        return best_move

    def _minimax(self, state, depth, alpha, beta, maximizing,
                 _parent_id: int = -1, _move_label: str = "?"):
        rec = self.tree_recorder
        node_id = -1
        if rec:
            node_type = "MAX" if maximizing else "MIN"
            node_id = rec.add_node(
                _parent_id if _parent_id >= 0 else None,
                DEPTH - depth,
                node_type,
                _move_label,
                alpha if alpha != float('-inf') else None,
                beta  if beta  != float('inf')  else None,
            )

        if self.profiler:
            self.profiler.count_node()
            self.profiler.update_depth(DEPTH - depth)

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
                                _parent_id=node_id, _move_label="pass")
            if rec and node_id >= 0:
                rec.update_value(node_id, val)
            return val

        if maximizing:
            val = float('-inf')
            for i, (tile, side) in enumerate(moves):
                ns = state.apply_move(tile, side, player)
                val = max(val, self._minimax(ns, depth - 1, alpha, beta, False,
                                             _parent_id=node_id,
                                             _move_label=f"{tile}\u2192{side}"))
                alpha = max(alpha, val)
                if beta <= alpha:
                    remaining = len(moves) - i - 1
                    if rec and node_id >= 0 and remaining > 0:
                        rec.add_node(node_id, DEPTH - depth + 1, "PRUNED",
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
                                             _parent_id=node_id,
                                             _move_label=f"{tile}\u2192{side}"))
                beta = min(beta, val)
                if beta <= alpha:
                    remaining = len(moves) - i - 1
                    if rec and node_id >= 0 and remaining > 0:
                        rec.add_node(node_id, DEPTH - depth + 1, "PRUNED",
                                     f"\u2702 {remaining} podado(s)",
                                     alpha, beta, pruned=True)
                    break
            if rec and node_id >= 0:
                rec.update_value(node_id, val)
            return val

    def _heuristic(self, state) -> float:
        m = manhattan_distance(state, self.player)
        e = euclidean_distance(state, self.player)
        return 0.5 * m + 0.5 * e
