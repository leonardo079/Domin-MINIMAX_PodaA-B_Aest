import random
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


# ──────────────────────────────────────────────────────────────────────────────
#  Tile: ficha de dominó. Valor canónico: min(a,b) | max(a,b).
#  En la mano se guarda en forma canónica.
#  En el tablero se guarda como OrientedTile para respetar la cadena visual.
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Tile:
    a: int
    b: int

    def fits(self, value: int) -> bool:
        return self.a == value or self.b == value

    def oriented(self, connect_value: int) -> Tuple[int, int]:
        """
        Retorna (conectado, libre) para encajar con connect_value.
        Ejemplo: Tile(3,5).oriented(3) → (3, 5)  → libre = 5
                 Tile(3,5).oriented(5) → (5, 3)  → libre = 3
        """
        if self.a == connect_value:
            return (self.a, self.b)
        return (self.b, self.a)

    def pips(self) -> int:
        return self.a + self.b

    def __repr__(self):
        return f"[{self.a}|{self.b}]"

    def __eq__(self, other):
        return (self.a == other.a and self.b == other.b) or \
               (self.a == other.b and self.b == other.a)

    def __hash__(self):
        return hash((min(self.a, self.b), max(self.a, self.b)))


class OrientedTile:
    """
    Ficha orientada para el tablero.
    left_val es el valor que queda hacia la izquierda del tablero,
    right_val el que queda hacia la derecha.
    Esto permite imprimir la cadena con la orientación correcta.
    """
    def __init__(self, left_val: int, right_val: int):
        self.left_val  = left_val
        self.right_val = right_val

    def as_tile(self) -> Tile:
        return Tile(min(self.left_val, self.right_val),
                    max(self.left_val, self.right_val))

    def __repr__(self):
        return f"[{self.left_val}|{self.right_val}]"

    def __eq__(self, other):
        if isinstance(other, OrientedTile):
            return ({self.left_val, self.right_val} ==
                    {other.left_val, other.right_val})
        if isinstance(other, Tile):
            return ({self.left_val, self.right_val} ==
                    {other.a, other.b})
        return False

    def __hash__(self):
        return hash((min(self.left_val, self.right_val),
                     max(self.left_val, self.right_val)))


def generate_all_tiles() -> list:
    return [Tile(a, b) for a in range(7) for b in range(a, 7)]


# ──────────────────────────────────────────────────────────────────────────────
#  GameState
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GameState:
    # El tablero almacena OrientedTile para que la cadena visual sea correcta:
    #   board[0].left_val  == left_end   (extremo izquierdo visible)
    #   board[-1].right_val == right_end  (extremo derecho visible)
    board: list = field(default_factory=list)
    left_end:  Optional[int] = None
    right_end: Optional[int] = None

    agent_hand:    list = field(default_factory=list)
    opponent_hand: list = field(default_factory=list)
    pool:          list = field(default_factory=list)

    current_player: int = 0
    pass_count:     int = 0
    history:        list = field(default_factory=list)
    agent_passes:   list = field(default_factory=list)
    opponent_passes:list = field(default_factory=list)

    # Fichas no visibles para el agente (mano oponente + pozo)
    unknown_tiles: list = field(default_factory=list)

    # ── Inicio de partida ──────────────────────────────────────────────────

    @classmethod
    def new_game(cls):
        tiles = generate_all_tiles()
        random.shuffle(tiles)
        agent_hand    = tiles[:7]
        opponent_hand = tiles[7:14]
        pool          = tiles[14:]

        # ── Regla oficial de inicio ──────────────────────────────────────
        # El jugador con el doble más alto EMPIEZA y coloca ese doble.
        # Si nadie tiene doble, empieza quien tenga la ficha de mayor peso.
        agent_doubles = [t for t in agent_hand if t.a == t.b]
        opp_doubles   = [t for t in opponent_hand if t.a == t.b]

        agent_best_double = max(agent_doubles, key=lambda t: t.a, default=None)
        opp_best_double   = max(opp_doubles,   key=lambda t: t.a, default=None)

        if agent_best_double and opp_best_double:
            if agent_best_double.a >= opp_best_double.a:
                first        = 0
                opening_tile = agent_best_double
                agent_hand   = [t for t in agent_hand if t != opening_tile]
            else:
                first        = 1
                opening_tile = opp_best_double
                opponent_hand = [t for t in opponent_hand if t != opening_tile]
        elif agent_best_double:
            first        = 0
            opening_tile = agent_best_double
            agent_hand   = [t for t in agent_hand if t != opening_tile]
        elif opp_best_double:
            first        = 1
            opening_tile = opp_best_double
            opponent_hand = [t for t in opponent_hand if t != opening_tile]
        else:
            # Nadie tiene doble: la ficha más pesada de cualquier mano abre
            all_hand = agent_hand + opponent_hand
            opening_tile = max(all_hand, key=lambda t: t.pips())
            if opening_tile in agent_hand:
                first     = 0
                agent_hand = [t for t in agent_hand if t != opening_tile]
            else:
                first         = 1
                opponent_hand = [t for t in opponent_hand if t != opening_tile]

        # La ficha de apertura se orienta canónicamente (left=a, right=b, a<=b)
        ot = OrientedTile(opening_tile.a, opening_tile.b)
        board     = [ot]
        left_end  = ot.left_val
        right_end = ot.right_val

        # El siguiente turno es del OTRO jugador (quien abrió ya jugó)
        current = 1 - first

        unknown = list(opponent_hand) + list(pool)

        return cls(
            board=board,
            left_end=left_end,
            right_end=right_end,
            agent_hand=agent_hand,
            opponent_hand=opponent_hand,
            pool=pool,
            current_player=current,
            unknown_tiles=unknown
        )

    # ── Probabilidades desde perspectiva del agente ────────────────────────

    def pool_size(self) -> int:
        return len(self.pool)

    def opponent_hand_size(self) -> int:
        return len(self.opponent_hand)

    def unknown_size(self) -> int:
        return len(self.unknown_tiles)

    def prob_tile_in_pool(self, tile: Tile) -> float:
        if tile in self.agent_hand:
            return 0.0
        total_unknown = self.unknown_size()
        if total_unknown == 0:
            return 0.0
        return self.pool_size() / total_unknown

    def prob_tile_in_opponent(self, tile: Tile) -> float:
        if tile in self.agent_hand:
            return 0.0
        total_unknown = self.unknown_size()
        if total_unknown == 0:
            return 0.0
        opp_size = self.opponent_hand_size()
        penalty = 1.0
        if self.opponent_passes:
            passed_at = set(self.opponent_passes)
            if tile.a in passed_at or tile.b in passed_at:
                penalty = 0.3
        return min(1.0, (opp_size / total_unknown) * penalty)

    def expected_pool_fits(self, end_value: int) -> float:
        if self.pool_size() == 0:
            return 0.0
        total_unknown = self.unknown_size()
        if total_unknown == 0:
            return 0.0
        fitting = sum(1 for t in self.unknown_tiles if t.fits(end_value))
        return fitting * (self.pool_size() / total_unknown)

    def draw_from_pool(self, player: int) -> Optional[Tile]:
        if not self.pool:
            return None
        tile = self.pool.pop(0)
        if player == 0:
            self.agent_hand.append(tile)
            if tile in self.unknown_tiles:
                self.unknown_tiles.remove(tile)
        else:
            self.opponent_hand.append(tile)
        return tile

    # ── Lógica de jugadas ──────────────────────────────────────────────────

    def valid_moves(self, hand: list) -> list:
        """
        Retorna lista de (Tile, 'left'|'right') válidos.

        - Si left_end == right_end: la ficha puede ir a ambos lados (se ofrecen
          ambas opciones para que apply_move actualice el extremo correcto).
        - Si left_end != right_end: cada lado es independiente.
        """
        if self.left_end is None:
            return [(t, 'left') for t in hand]

        moves = []
        for tile in hand:
            fits_left  = tile.fits(self.left_end)
            fits_right = tile.fits(self.right_end)

            if self.left_end == self.right_end:
                if fits_left:
                    moves.append((tile, 'left'))
                    moves.append((tile, 'right'))
            else:
                if fits_left:
                    moves.append((tile, 'left'))
                if fits_right:
                    moves.append((tile, 'right'))
        return moves

    def apply_move(self, tile: Tile, side: str, player: int) -> 'GameState':
        """
        Aplica el movimiento y retorna el nuevo estado.

        La cadena del tablero se mantiene orientada:
          board[0].left_val  == left_end
          board[-1].right_val == right_end

        Ejemplo (del enunciado):
          Tablero: [1|3], extremos 1 ... 3
          Jugar [3|4] en right:
            oriented(3) → conectado=3, libre=4
            OrientedTile(3, 4) → tablero: [1|3][3|4], right_end=4  ✓
          Jugar [1|6] en left:
            oriented(1) → conectado=1, libre=6
            OrientedTile(6, 1) → tablero: [6|1][1|3][3|4], left_end=6  ✓
        """
        import copy
        ns = copy.deepcopy(self)
        hand = ns.agent_hand if player == 0 else ns.opponent_hand
        hand[:] = [t for t in hand if not (t == tile)]
        ns.unknown_tiles = [t for t in ns.unknown_tiles if not (t == tile)]

        if ns.left_end is None:
            # Salvaguarda (no ocurre con new_game correcto)
            ot = OrientedTile(tile.a, tile.b)
            ns.board.append(ot)
            ns.left_end  = ot.left_val
            ns.right_end = ot.right_val

        elif side == 'left':
            # La ficha se conecta por su valor == left_end
            # El valor libre queda a la IZQUIERDA (nuevo left_end)
            connected, free = tile.oriented(ns.left_end)
            ot = OrientedTile(free, connected)   # free | connected → left_end = free
            ns.board.insert(0, ot)
            ns.left_end = free

        else:  # side == 'right'
            # La ficha se conecta por su valor == right_end
            # El valor libre queda a la DERECHA (nuevo right_end)
            connected, free = tile.oriented(ns.right_end)
            ot = OrientedTile(connected, free)   # connected | free → right_end = free
            ns.board.append(ot)
            ns.right_end = free

        ns.pass_count = 0
        ns.history.append({'player': player, 'tile': tile, 'side': side})
        ns.current_player = 1 - player
        return ns

    def apply_draw_and_play(self, player: int):
        """
        Roba del pozo hasta poder jugar o agotar el pozo.
        Retorna (nuevo_estado, jugadas_disponibles).
        """
        import copy
        ns = copy.deepcopy(self)

        while ns.pool:
            tile = ns.pool.pop(0)
            if player == 0:
                ns.agent_hand.append(tile)
                if tile in ns.unknown_tiles:
                    ns.unknown_tiles.remove(tile)
            else:
                ns.opponent_hand.append(tile)
            ns.history.append({'player': player, 'tile': None, 'side': 'draw'})

            current_hand = ns.agent_hand if player == 0 else ns.opponent_hand
            moves = ns.valid_moves(current_hand)
            if moves:
                return ns, moves

        return ns, []

    def apply_pass(self, player: int) -> 'GameState':
        import copy
        ns = copy.deepcopy(self)
        ns.pass_count += 1
        ns.history.append({'player': player, 'tile': None, 'side': None})
        if player == 0:
            ns.agent_passes.append(ns.left_end)
            ns.agent_passes.append(ns.right_end)
        else:
            ns.opponent_passes.append(ns.left_end)
            ns.opponent_passes.append(ns.right_end)
        ns.current_player = 1 - player
        return ns

    # ── Estado terminal y ganador ──────────────────────────────────────────

    def is_terminal(self) -> bool:
        return (len(self.agent_hand) == 0 or
                len(self.opponent_hand) == 0 or
                self.pass_count >= 2)

    def winner(self) -> Optional[int]:
        if not self.is_terminal():
            return None
        if len(self.agent_hand) == 0:
            return 0
        if len(self.opponent_hand) == 0:
            return 1
        agent_pips = sum(t.pips() for t in self.agent_hand)
        opp_pips   = sum(t.pips() for t in self.opponent_hand)
        if agent_pips < opp_pips:
            return 0
        elif opp_pips < agent_pips:
            return 1
        return -1

    def pip_sum(self, player: int) -> int:
        hand = self.agent_hand if player == 0 else self.opponent_hand
        return sum(t.pips() for t in hand)

    def board_str(self) -> str:
        """Cadena visual del tablero con orientación correcta."""
        return " ".join(str(ot) for ot in self.board)