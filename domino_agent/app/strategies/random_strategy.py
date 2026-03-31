"""
random_strategy.py — Baseline: jugada aleatoria entre las válidas.
"""
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

        rec = self.tree_recorder
        if rec:
            rec.reset()

        hand = state.agent_hand if self.player == 0 else state.opponent_hand
        moves = state.valid_moves(hand)

        root_id = -1
        if rec:
            root_id = rec.add_node(
                None, 0, "ROOT",
                f"Turno jugador {self.player} (Random)",
                None, None,
            )
            for tile, side in moves:
                rec.add_node(root_id, 1, "RANDOM", f"{tile}→{side}", None, None)

        if not moves:
            if self.profiler:
                self.profiler.end_turn("pass")
            return None

        chosen = random.choice(moves)

        if rec:
            label = f"{chosen[0]}→{chosen[1]}"
            for node in rec._nodes:
                if node["node_type"] == "RANDOM" and node["move"] == label:
                    node["value"] = 1.0
                    break

        if self.profiler:
            self.profiler.count_node()
            self.profiler.end_turn(str(chosen))

        return chosen