import heapq
from typing import Optional, Tuple
from game_state import GameState, Tile
from strategies.base import AgentStrategy
from evaluator import manhattan_distance, euclidean_distance

MAX_NODES = 2000


class AStarStrategy(AgentStrategy):

    @property
    def name(self) -> str:
        return "astar"

    def decide(self, state: GameState) -> Optional[Tuple[Tile, str]]:
        if self.profiler:
            self.profiler.start_turn()

        moves = state.valid_moves(state.agent_hand)
        if not moves:
            if self.profiler:
                self.profiler.end_turn("pass")
            return None

        # Heap: (f, g, id, state, first_move)
        counter = 0
        initial_g = 7 - len(state.agent_hand)  # fichas ya colocadas
        heap = []

        for tile, side in moves:
            ns = state.apply_move(tile, side, self.player)
            g = initial_g + 1
            h = self._heuristic(ns)
            f = g + h
            heapq.heappush(heap, (f, g, counter, ns, (tile, side)))
            counter += 1
            if self.profiler:
                self.profiler.count_node()

        best_move = moves[0]
        best_f = float('inf')
        nodes_explored = 0

        while heap and nodes_explored < MAX_NODES:
            f, g, _, current_state, first_move = heapq.heappop(heap)
            nodes_explored += 1

            if self.profiler:
                self.profiler.count_eval()
                self.profiler.update_depth(g - initial_g)

            if f < best_f:
                best_f = f
                best_move = first_move

            if current_state.is_terminal():
                break

            # Expandir solo jugadas del agente (A* no modela oponente)
            next_moves = current_state.valid_moves(current_state.agent_hand)
            for tile, side in next_moves:
                ns2 = current_state.apply_move(tile, side, self.player)
                # Simula turno oponente con jugada aleatoria para avanzar estado
                opp_moves = ns2.valid_moves(ns2.opponent_hand)
                if opp_moves:
                    import random
                    ot, os_ = random.choice(opp_moves)
                    ns2 = ns2.apply_move(ot, os_, 1 - self.player)
                new_g = g + 1
                new_h = self._heuristic(ns2)
                new_f = new_g + new_h
                heapq.heappush(heap, (new_f, new_g, counter, ns2, first_move))
                counter += 1
                if self.profiler:
                    self.profiler.count_node()

        if self.profiler:
            self.profiler.end_turn(str(best_move))
        return best_move

    def _heuristic(self, state) -> float:
        """h(n) combinando Manhattan y Euclidiana."""
        m = manhattan_distance(state, self.player)
        e = euclidean_distance(state, self.player)
        return (m + e) / 2.0