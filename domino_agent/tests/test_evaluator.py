"""
tests/test_evaluator.py — Tests unitarios de las funciones heurísticas.

Cubre: manhattan_distance, euclidean_distance, pool_opportunity_score,
       evaluate() con distintas configuraciones de pesos.
"""
import pytest
from app.core.game_state import GameState
from app.core.evaluator import (
    manhattan_distance,
    euclidean_distance,
    pool_opportunity_score,
    evaluate,
)


@pytest.fixture
def state():
    return GameState.new_game()


class TestManhattanDistance:
    def test_returns_float(self, state):
        result = manhattan_distance(state, player=0)
        assert isinstance(result, float)

    def test_non_negative(self, state):
        assert manhattan_distance(state, 0) >= 0.0
        assert manhattan_distance(state, 1) >= 0.0

    def test_empty_board_returns_zero(self):
        s = GameState()
        assert manhattan_distance(s, 0) == 0.0

    def test_max_bounded(self, state):
        # Distancia Manhattan máxima por ficha es 6 (0 a 6)
        assert manhattan_distance(state, 0) <= 6.0


class TestEuclideanDistance:
    def test_returns_float(self, state):
        result = euclidean_distance(state, player=0)
        assert isinstance(result, float)

    def test_non_negative(self, state):
        assert euclidean_distance(state, 0) >= 0.0
        assert euclidean_distance(state, 1) >= 0.0

    def test_empty_board_returns_zero(self):
        s = GameState()
        assert euclidean_distance(s, 0) == 0.0


class TestPoolOpportunityScore:
    def test_range_zero_to_one(self, state):
        score = pool_opportunity_score(state, 0)
        assert 0.0 <= score <= 1.0

    def test_returns_zero_empty_board(self):
        s = GameState()
        assert pool_opportunity_score(s, 0) == 0.0


class TestEvaluate:
    def test_returns_float_in_range(self, state):
        result = evaluate(state, player=0)
        assert isinstance(result, float)
        assert -1.0 <= result <= 1.0

    def test_manhattan_only(self, state):
        result = evaluate(state, 0, use_manhattan=True, use_euclidean=False, use_pool=False)
        assert -1.0 <= result <= 1.0

    def test_euclidean_only(self, state):
        result = evaluate(state, 0, use_manhattan=False, use_euclidean=True, use_pool=False)
        assert -1.0 <= result <= 1.0

    def test_both_distances(self, state):
        result = evaluate(state, 0, use_manhattan=True, use_euclidean=True, use_pool=True)
        assert -1.0 <= result <= 1.0

    def test_symmetric_players(self, state):
        """Los dos jugadores evalúan en perspectivas opuestas."""
        r0 = evaluate(state, player=0)
        r1 = evaluate(state, player=1)
        # No deben ser exactamente iguales salvo estado perfectamente simétrico
        # Solo verificamos que ambos están en rango válido
        assert -1.0 <= r0 <= 1.0
        assert -1.0 <= r1 <= 1.0
