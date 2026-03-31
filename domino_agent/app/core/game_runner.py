"""
game_runner.py — Función compartida para ejecutar una partida completa.

Centraliza el bucle "robar → decidir → aplicar" que antes estaba duplicado en:
  - benchmark.py  (_play_single_game)
  - test_strategies.py  (test_full_game_completes / test_matchup_game_completes)

La función es síncrona y no tiene dependencias de FastAPI; puede usarse tanto
desde los endpoints como desde los tests sin adaptaciones.
"""
from typing import Tuple, Optional

from app.core.game_state import GameState
from app.core.profiler import CostProfiler
from app.strategies.base import AgentStrategy


def play_full_game(
    sa: AgentStrategy,
    sb: AgentStrategy,
    prof_a: Optional[CostProfiler] = None,
    prof_b: Optional[CostProfiler] = None,
    max_turns: int = 200,
) -> Tuple[Optional[int], int, int, int]:
    """
    Ejecuta una partida completa entre dos agentes ya instanciados.

    Los profilers son opcionales; si se pasan, deben estar ya asociados
    a las estrategias (sa.set_profiler / sb.set_profiler).

    Retorna:
        (winner, turn_count, pip_sum_a, pip_sum_b)

        winner → 0 | 1 | -1 (empate) | None (límite de turnos alcanzado)
    """
    state = GameState.new_game()
    turn = 0

    while not state.is_terminal() and turn < max_turns:
        turn += 1
        cur = state.current_player
        strat = sa if cur == 0 else sb
        hand = state.agent_hand if cur == 0 else state.opponent_hand

        # Robar del pozo hasta tener jugada o vaciarlo
        moves = state.valid_moves(hand)
        if not moves and state.pool:
            state, moves = state.apply_draw_and_play(cur)

        result = strat.decide(state)
        if result is None:
            state = state.apply_pass(cur)
        else:
            tile, side = result
            state = state.apply_move(tile, side, cur)

    return state.winner(), turn, state.pip_sum(0), state.pip_sum(1)