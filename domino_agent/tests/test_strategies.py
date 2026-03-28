"""
tests/test_strategies.py — Tests unitarios de las estrategias de IA.

Verifica que cada estrategia:
  - Retorna (Tile, side) válido o None
  - No muta el estado original
  - Registra métricas en el profiler
  - Completa una partida completa sin errores
"""
import pytest
from app.core.game_state import GameState, Tile
from app.core.profiler import CostProfiler
from app.strategies import STRATEGIES


ALL_STRATEGIES = list(STRATEGIES.keys())


@pytest.fixture
def fresh_state():
    return GameState.new_game()


@pytest.mark.parametrize("strategy_name", ALL_STRATEGIES)
class TestStrategyDecide:

    def _make_strategy(self, name: str, player: int = 0):
        strat = STRATEGIES[name](player=player)
        prof = CostProfiler(name)
        strat.set_profiler(prof)
        return strat, prof

    def test_decide_returns_valid_type(self, strategy_name, fresh_state):
        strat, _ = self._make_strategy(strategy_name, player=0)
        state = fresh_state
        # Forzamos turno del jugador 0 si no es su turno
        import copy
        s = copy.deepcopy(state)
        s.current_player = 0
        result = strat.decide(s)
        assert result is None or (
            isinstance(result, tuple) and
            len(result) == 2 and
            isinstance(result[0], Tile) and
            result[1] in ('left', 'right')
        )

    def test_decide_does_not_mutate_state(self, strategy_name, fresh_state):
        strat, _ = self._make_strategy(strategy_name, player=0)
        import copy
        s = copy.deepcopy(fresh_state)
        s.current_player = 0
        board_before = len(s.board)
        hand_before = len(s.agent_hand)
        strat.decide(s)
        assert len(s.board) == board_before
        assert len(s.agent_hand) == hand_before

    def test_decide_records_profiler_metrics(self, strategy_name, fresh_state):
        strat, prof = self._make_strategy(strategy_name, player=0)
        import copy
        s = copy.deepcopy(fresh_state)
        s.current_player = 0
        strat.decide(s)
        assert len(prof.metrics) == 1

    def test_move_is_in_valid_moves(self, strategy_name, fresh_state):
        strat, _ = self._make_strategy(strategy_name, player=0)
        import copy
        s = copy.deepcopy(fresh_state)
        s.current_player = 0
        valid = s.valid_moves(s.agent_hand)
        result = strat.decide(s)
        if result is not None and valid:
            assert result in valid


@pytest.mark.parametrize("strategy_name", ALL_STRATEGIES)
def test_full_game_completes(strategy_name):
    """Una partida completa con la misma estrategia en ambos lados no debe colgarse."""
    state = GameState.new_game()
    sa = STRATEGIES[strategy_name](player=0)
    sb = STRATEGIES[strategy_name](player=1)
    sa.set_profiler(CostProfiler(strategy_name))
    sb.set_profiler(CostProfiler(strategy_name))

    turn = 0
    max_turns = 200  # Salvaguarda contra bucles infinitos

    while not state.is_terminal() and turn < max_turns:
        turn += 1
        cur = state.current_player
        strat = sa if cur == 0 else sb
        hand = state.agent_hand if cur == 0 else state.opponent_hand
        moves = state.valid_moves(hand)

        if not moves and state.pool:
            state, moves = state.apply_draw_and_play(cur)

        result = strat.decide(state)
        if result is None:
            state = state.apply_pass(cur)
        else:
            tile, side = result
            state = state.apply_move(tile, side, cur)

    assert state.is_terminal() or turn < max_turns, (
        f"La estrategia '{strategy_name}' no terminó la partida en {max_turns} turnos"
    )


@pytest.mark.parametrize("name_a,name_b", [
    ("manhattan", "random"),
    ("euclidean", "random"),
    ("astar",     "random"),
    ("hybrid",    "manhattan"),
])
def test_matchup_game_completes(name_a, name_b):
    """Verifica que matchups cruzados completan la partida."""
    state = GameState.new_game()
    sa = STRATEGIES[name_a](player=0)
    sb = STRATEGIES[name_b](player=1)
    sa.set_profiler(CostProfiler(name_a))
    sb.set_profiler(CostProfiler(name_b))

    turn = 0
    while not state.is_terminal() and turn < 200:
        turn += 1
        cur = state.current_player
        strat = sa if cur == 0 else sb
        hand = state.agent_hand if cur == 0 else state.opponent_hand
        moves = state.valid_moves(hand)

        if not moves and state.pool:
            state, moves = state.apply_draw_and_play(cur)

        result = strat.decide(state)
        if result is None:
            state = state.apply_pass(cur)
        else:
            tile, side = result
            state = state.apply_move(tile, side, cur)

    assert state.is_terminal()
    assert state.winner() in (0, 1, -1)
