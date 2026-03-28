from app.strategies.random_strategy import RandomStrategy
from app.strategies.manhattan_strategy import ManhattanStrategy
from app.strategies.euclidean_strategy import EuclideanStrategy
from app.strategies.astar_strategy import AStarStrategy
from app.strategies.hybrid_strategy import HybridStrategy

STRATEGIES = {
    "random": RandomStrategy,
    "manhattan": ManhattanStrategy,
    "euclidean": EuclideanStrategy,
    "astar": AStarStrategy,
    "hybrid": HybridStrategy,
}

STRATEGY_DESCRIPTIONS = {
    "random": "Jugada aleatoria entre las válidas (baseline)",
    "manhattan": "Minimax + poda α-β con distancia Manhattan",
    "euclidean": "Minimax + poda α-β con distancia Euclidiana",
    "astar": "Búsqueda A* pura (Manhattan + Euclidiana como heurística)",
    "hybrid": "A* + Minimax + α-β + profundidad adaptativa + pozo",
}
