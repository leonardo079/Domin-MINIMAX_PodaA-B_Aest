"""
tests/test_game_state.py — Tests unitarios del núcleo del juego.

Cubre: generación de fichas, inicio de partida, jugadas válidas,
       apply_move, apply_pass, pozo, terminal y ganador.
"""
import pytest
from app.core.game_state import (
    GameState, Tile, OrientedTile, generate_all_tiles
)


# ── Tile ───────────────────────────────────────────────────────────────────────

class TestTile:
    def test_generate_all_tiles_count(self):
        tiles = generate_all_tiles()
        assert len(tiles) == 28

    def test_tile_pips(self):
        t = Tile(3, 5)
        assert t.pips() == 8

    def test_tile_fits(self):
        t = Tile(3, 5)
        assert t.fits(3)
        assert t.fits(5)
        assert not t.fits(2)

    def test_tile_oriented_left(self):
        t = Tile(3, 5)
        connected, free = t.oriented(3)
        assert connected == 3 and free == 5

    def test_tile_oriented_right(self):
        t = Tile(3, 5)
        connected, free = t.oriented(5)
        assert connected == 5 and free == 3

    def test_tile_equality(self):
        assert Tile(3, 5) == Tile(5, 3)
        assert Tile(3, 5) != Tile(3, 4)

    def test_tile_hash_symmetric(self):
        assert hash(Tile(2, 6)) == hash(Tile(6, 2))


# ── GameState.new_game ─────────────────────────────────────────────────────────

class TestNewGame:
    def test_hands_size(self, game_state):
        # La ficha de apertura sale de una mano, así que una tiene 6 y la otra 7
        total = len(game_state.agent_hand) + len(game_state.opponent_hand)
        assert total == 13

    def test_pool_size(self, game_state):
        # 28 fichas totales - 1 apertura (sale de una mano de 7 → queda 6)
        # - 6 en mano A - 7 en mano B = 14  (o bien 7-6-1 = 14)
        assert game_state.pool_size() == 14

    def test_board_starts_with_one_tile(self, game_state):
        assert len(game_state.board) == 1

    def test_board_ends_consistent(self, game_state):
        assert game_state.left_end == game_state.board[0].left_val
        assert game_state.right_end == game_state.board[-1].right_val

    def test_total_tiles_accounted(self, game_state):
        total = (len(game_state.agent_hand) +
                 len(game_state.opponent_hand) +
                 game_state.pool_size() +
                 len(game_state.board))
        assert total == 28

    def test_no_duplicate_tiles(self, game_state):
        all_tiles = (
            list(game_state.agent_hand)
            + list(game_state.opponent_hand)
            + list(game_state.pool)
            + [ot.as_tile() for ot in game_state.board]
        )
        assert len(all_tiles) == len(set(all_tiles))


# ── valid_moves ────────────────────────────────────────────────────────────────

class TestValidMoves:
    def test_returns_list(self, game_state):
        hand = game_state.agent_hand
        moves = game_state.valid_moves(hand)
        assert isinstance(moves, list)

    def test_move_format(self, game_state):
        hand = game_state.agent_hand
        moves = game_state.valid_moves(hand)
        for tile, side in moves:
            assert isinstance(tile, Tile)
            assert side in ('left', 'right')

    def test_move_tiles_fit_board(self, game_state):
        hand = game_state.agent_hand
        moves = game_state.valid_moves(hand)
        for tile, side in moves:
            if side == 'left':
                assert tile.fits(game_state.left_end)
            else:
                assert tile.fits(game_state.right_end)


# ── apply_move ─────────────────────────────────────────────────────────────────

class TestApplyMove:
    def _first_valid_move(self, state, player):
        hand = state.agent_hand if player == 0 else state.opponent_hand
        moves = state.valid_moves(hand)
        return moves[0] if moves else None

    def test_hand_shrinks_after_move(self, game_state):
        player = game_state.current_player
        move = self._first_valid_move(game_state, player)
        if move is None:
            pytest.skip("Sin jugadas disponibles en este estado")
        tile, side = move
        before = len(game_state.agent_hand if player == 0 else game_state.opponent_hand)
        ns = game_state.apply_move(tile, side, player)
        after = len(ns.agent_hand if player == 0 else ns.opponent_hand)
        assert after == before - 1

    def test_board_grows_after_move(self, game_state):
        player = game_state.current_player
        move = self._first_valid_move(game_state, player)
        if move is None:
            pytest.skip("Sin jugadas disponibles en este estado")
        tile, side = move
        before = len(game_state.board)
        ns = game_state.apply_move(tile, side, player)
        assert len(ns.board) == before + 1

    def test_player_switches_after_move(self, game_state):
        player = game_state.current_player
        move = self._first_valid_move(game_state, player)
        if move is None:
            pytest.skip("Sin jugadas disponibles")
        tile, side = move
        ns = game_state.apply_move(tile, side, player)
        assert ns.current_player == 1 - player

    def test_board_ends_consistent_after_move(self, game_state):
        player = game_state.current_player
        move = self._first_valid_move(game_state, player)
        if move is None:
            pytest.skip("Sin jugadas disponibles")
        tile, side = move
        ns = game_state.apply_move(tile, side, player)
        assert ns.left_end == ns.board[0].left_val
        assert ns.right_end == ns.board[-1].right_val

    def test_original_state_not_mutated(self, game_state):
        player = game_state.current_player
        move = self._first_valid_move(game_state, player)
        if move is None:
            pytest.skip("Sin jugadas disponibles")
        original_board_len = len(game_state.board)
        tile, side = move
        game_state.apply_move(tile, side, player)
        assert len(game_state.board) == original_board_len


# ── apply_pass ─────────────────────────────────────────────────────────────────

class TestApplyPass:
    def test_pass_increments_pass_count(self, game_state):
        ns = game_state.apply_pass(game_state.current_player)
        assert ns.pass_count == game_state.pass_count + 1

    def test_pass_switches_player(self, game_state):
        cur = game_state.current_player
        ns = game_state.apply_pass(cur)
        assert ns.current_player == 1 - cur

    def test_two_passes_terminal(self, game_state):
        ns = game_state.apply_pass(0)
        ns = ns.apply_pass(1)
        assert ns.is_terminal()


# ── pip_sum & winner ───────────────────────────────────────────────────────────

class TestTerminal:
    def test_pip_sum_non_negative(self, game_state):
        assert game_state.pip_sum(0) >= 0
        assert game_state.pip_sum(1) >= 0

    def test_not_terminal_at_start(self, game_state):
        assert not game_state.is_terminal()

    def test_winner_none_when_not_terminal(self, game_state):
        assert game_state.winner() is None
