from abc import ABC, abstractmethod
from typing import Optional, Tuple
from game_state import GameState, Tile
from profiler import CostProfiler


class AgentStrategy(ABC):
    """
    Interfaz que toda estrategia debe implementar.
    decide() recibe el estado actual y retorna (ficha, extremo) o None si pasa.
    """

    def __init__(self, player: int = 0):
        self.player = player
        self.profiler: Optional[CostProfiler] = None

    def set_profiler(self, profiler: CostProfiler):
        self.profiler = profiler

    @abstractmethod
    def decide(self, state: GameState) -> Optional[Tuple[Tile, str]]:
        """
        Retorna (Tile, 'left'|'right') o None si el agente pasa.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass