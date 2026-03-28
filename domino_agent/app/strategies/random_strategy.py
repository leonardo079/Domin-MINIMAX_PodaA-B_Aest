import random
from typing import Optional, Tuple
from app.core.game_state import GameState, Tile
from app.strategies.base import AgentStrategy


class RandomStrategy(AgentStrategy):

    @property
    def name(self) -> str:
        return "random"

    def decide(self, state: GameState) -> Optional[Tuple[Tile, str]]:
        if self.profiler:
            self.profiler.start_turn()

        moves = state.valid_moves(state.agent_hand if self.player == 0 else state.opponent_hand)

        if self.profiler:
            self.profiler.count_node()
            move_str = str(moves[0]) if moves else "pass"
            self.profiler.end_turn(move_str)

        if not moves:
            return None
        return random.choice(moves)
