"""
manhattan_strategy.py — Minimax + poda alpha-beta con heurística Manhattan.
"""
from app.strategies.base import MinimaxStrategy


class ManhattanStrategy(MinimaxStrategy):
    use_manhattan = True
    use_euclidean = False
    use_pool = False
    depth = 5

    @property
    def name(self) -> str:
        return "manhattan"