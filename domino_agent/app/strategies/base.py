from abc import ABC, abstractmethod
from typing import Optional, Tuple
from app.core.game_state import GameState, Tile
from app.core.profiler import CostProfiler


class AgentStrategy(ABC):
    """Interfaz que toda estrategia debe implementar."""

    def __init__(self, player: int = 0):
        self.player = player
        self.profiler: Optional[CostProfiler] = None
        self.tree_recorder = None   # instancia de TreeRecorder, opcional

    def set_profiler(self, profiler: CostProfiler):
        self.profiler = profiler

    def set_tree_recorder(self, recorder) -> None:
        """Asocia un TreeRecorder para registrar el árbol de búsqueda."""
        self.tree_recorder = recorder

    @abstractmethod
    def decide(self, state: GameState) -> Optional[Tuple[Tile, str]]:
        """Retorna (Tile, 'left'|'right') o None si el agente pasa."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
