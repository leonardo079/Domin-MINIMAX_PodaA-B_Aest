from strategies.random_strategy import RandomStrategy
from strategies.manhattan_strategy import ManhattanStrategy
from strategies.euclidean_strategy import EuclideanStrategy
from strategies.astar_strategy import AStarStrategy
from strategies.hybrid_strategy import HybridStrategy

STRATEGIES = {
    'random':    RandomStrategy,
    'manhattan': ManhattanStrategy,
    'euclidean': EuclideanStrategy,
    'astar':     AStarStrategy,
    'hybrid':    HybridStrategy,
}