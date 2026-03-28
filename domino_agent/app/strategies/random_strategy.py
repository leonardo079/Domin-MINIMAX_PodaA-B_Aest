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
            root_id = rec.add_node(None, 0, "ROOT",
                                   f"Turno jugador {self.player} (Random)",
                                   None, None)
            for tile, side in moves:
                rec.add_node(root_id, 1, "RANDOM", f"{tile}\u2192{side}",
                             None, None)

        if self.profiler:
            self.profiler.count_node()
            move_str = str(moves[0]) if moves else "pass"
            self.profiler.end_turn(move_str)

        if not moves:
            return None

        chosen = random.choice(moves)

        # Marcar la jugada elegida con value=1.0
        if rec:
            label = f"{chosen[0]}\u2192{chosen[1]}"
            for node in rec._nodes:
                if node["node_type"] == "RANDOM" and node["move"] == label:
                    node["value"] = 1.0
                    break

        return chosen
