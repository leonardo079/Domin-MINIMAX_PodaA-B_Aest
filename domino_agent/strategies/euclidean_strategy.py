from typing import Optional, Tuple
from game_state import GameState, Tile
from strategies.base import AgentStrategy
from evaluator import evaluate

DEPTH = 4


class EuclideanStrategy(AgentStrategy):

    @property
    def name(self) -> str:
        return "euclidean"

    def decide(self, state: GameState) -> Optional[Tuple[Tile, str]]:
        if self.profiler:
            self.profiler.start_turn()

        moves = state.valid_moves(state.agent_hand)
        if not moves:
            if self.profiler:
                self.profiler.end_turn("pass")
            return None

        best_score = float('-inf')
        best_move = moves[0]

        for tile, side in moves:
            if self.profiler:
                self.profiler.count_node()
            ns = state.apply_move(tile, side, self.player)
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
                            use_manhattan=False, use_euclidean=True)

        hand = state.agent_hand if maximizing else state.opponent_hand
        player = self.player if maximizing else 1 - self.player
        moves = state.valid_moves(hand)

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