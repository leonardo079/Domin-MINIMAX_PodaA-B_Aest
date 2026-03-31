"""
euclidean_strategy.py — Minimax + poda alpha-beta con heurística Euclidiana.
"""
from app.strategies.base import MinimaxStrategy


class EuclideanStrategy(MinimaxStrategy):
    use_manhattan = False
    use_euclidean = True
    use_pool = False
    depth = 5

    @property
    def name(self) -> str:
        return "euclidean"