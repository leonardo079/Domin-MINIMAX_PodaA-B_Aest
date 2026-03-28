import heapq
import random
from typing import Optional, Tuple
from app.core.game_state import GameState, Tile
from app.strategies.base import AgentStrategy
from app.core.evaluator import manhattan_distance, euclidean_distance

MAX_NODES = 2000


class AStarStrategy(AgentStrategy):

    @property
    def name(self) -> str:
        return "astar"

    def decide(self, state: GameState) -> Optional[Tuple[Tile, str]]:
        if self.profiler:
            self.profiler.start_turn()

        rec = self.tree_recorder
        if rec:
            rec.reset()

        hand = state.agent_hand if self.player == 0 else state.opponent_hand
        moves = state.valid_moves(hand)
        if not moves:
            if self.profiler:
                self.profiler.end_turn("pass")
            return None

        # Nodo raíz
        root_id = -1
        if rec:
            root_id = rec.add_node(None, 0, "ROOT",
                                   f"Turno jugador {self.player} (A*)",
                                   None, None)

        counter = 0
        initial_g = 7 - len(hand)
        heap = []

        # heap tupla: (f, g, counter, state, first_move, node_id)
        for tile, side in moves:
            ns = state.apply_move(tile, side, self.player)
            g = initial_g + 1
            h = self._heuristic(ns)
            f = g + h
            node_id = -1
            if rec:
                node_id = rec.add_node(
                    root_id if root_id >= 0 else None,
                    1, "ASTAR",
                    f"{tile}\u2192{side}",
                    alpha=round(f, 4), beta=round(h, 4), value=round(g, 4),
                )
            heapq.heappush(heap, (f, g, counter, ns, (tile, side), node_id))
            counter += 1
            if self.profiler:
                self.profiler.count_node()

        best_move = moves[0]
        best_f = float('inf')
        nodes_explored = 0

        while heap and nodes_explored < MAX_NODES:
            f, g, _, current_state, first_move, current_node_id = heapq.heappop(heap)
            nodes_explored += 1

            if self.profiler:
                self.profiler.count_eval()
                self.profiler.update_depth(g - initial_g)

            if f < best_f:
                best_f = f
                best_move = first_move

            if current_state.is_terminal():
                break

            cur_hand = current_state.agent_hand if self.player == 0 else current_state.opponent_hand
            next_moves = current_state.valid_moves(cur_hand)
            for tile, side in next_moves:
                ns2 = current_state.apply_move(tile, side, self.player)
                opp_hand = ns2.opponent_hand if self.player == 0 else ns2.agent_hand
                opp_moves = ns2.valid_moves(opp_hand)
                if opp_moves:
                    ot, os_ = random.choice(opp_moves)
                    ns2 = ns2.apply_move(ot, os_, 1 - self.player)
                new_g = g + 1
                new_h = self._heuristic(ns2)
                new_f = new_g + new_h
                child_node_id = -1
                if rec:
                    child_node_id = rec.add_node(
                        current_node_id if current_node_id >= 0 else None,
                        g - initial_g + 1, "ASTAR",
                        f"{tile}\u2192{side}",
                        alpha=round(new_f, 4), beta=round(new_h, 4), value=round(new_g, 4),
                    )
                heapq.heappush(heap, (new_f, new_g, counter, ns2, first_move, child_node_id))
                counter += 1
                if self.profiler:
                    self.profiler.count_node()

        if self.profiler:
            self.profiler.end_turn(str(best_move))
        return best_move

    def _heuristic(self, state) -> float:
        m = manhattan_distance(state, self.player)
        e = euclidean_distance(state, self.player)
        return (m + e) / 2.0
