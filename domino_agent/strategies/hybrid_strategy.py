from typing import Optional, Tuple
from game_state import GameState, Tile
from strategies.base import AgentStrategy
from evaluator import evaluate, manhattan_distance, euclidean_distance

DEPTH = 4
TOP_K = 4  # A* filtra las K mejores jugadas antes de Minimax


class HybridStrategy(AgentStrategy):

    @property
    def name(self) -> str:
        return "hybrid"

    def decide(self, state: GameState) -> Optional[Tuple[Tile, str]]:
        if self.profiler:
            self.profiler.start_turn()

        moves = state.valid_moves(state.agent_hand)
        if not moves:
            if self.profiler:
                self.profiler.end_turn("pass")
            return None

        # A*: ordenar jugadas por heurística f(n) = g + h
        g = 7 - len(state.agent_hand)
        scored = []
        for tile, side in moves:
            ns = state.apply_move(tile, side, self.player)
            h = self._heuristic(ns)
            scored.append((g + h, (tile, side), ns))
            if self.profiler:
                self.profiler.count_node()

        scored.sort(key=lambda x: x[0])
        top_moves = [(m, ns) for _, m, ns in scored[:TOP_K]]

        # Minimax sobre las mejores candidatas
        best_score = float('-inf')
        best_move = top_moves[0][0]

        for (tile, side), ns in top_moves:
            if self.profiler:
                self.profiler.count_node()
            score = self._minimax(ns, DEPTH - 1, float('-inf'), float('inf'), False)
            if score > best_score:
                best_score = score
                best_move = (tile, side)

        if self.profiler:
            self.profiler.end_turn(str(best_move))
        return best_move

    def _minimax(self, state, depth, alpha, beta, maximizing):
        if self.profiler:
            self.profiler.count_node()
            self.profiler.update_depth(DEPTH - depth)

        if depth == 0 or state.is_terminal():
            if self.profiler:
                self.profiler.count_eval()
            return evaluate(state, self.player,
                            use_manhattan=True, use_euclidean=True)

        hand = state.agent_hand if maximizing else state.opponent_hand
        player = self.player if maximizing else 1 - self.player
        moves = state.valid_moves(hand)

        # A* ordering dentro de Minimax también
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
            return self._minimax(ns, depth - 1, alpha, beta, not maximizing)

        if maximizing:
            val = float('-inf')
            for tile, side in moves:
                ns = state.apply_move(tile, side, player)
                val = max(val, self._minimax(ns, depth - 1, alpha, beta, False))
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
            return val
        else:
            val = float('inf')
            for tile, side in moves:
                ns = state.apply_move(tile, side, player)
                val = min(val, self._minimax(ns, depth - 1, alpha, beta, True))
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return val

    def _heuristic(self, state) -> float:
        m = manhattan_distance(state, self.player)
        e = euclidean_distance(state, self.player)
        return 0.5 * m + 0.5 * e