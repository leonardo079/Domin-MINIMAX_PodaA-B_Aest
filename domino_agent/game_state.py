import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Tile:
    a: int
    b: int

    def fits(self, value: int) -> bool:
        return self.a == value or self.b == value

    def oriented(self, value: int) -> tuple:
        """Retorna (valor_conectado, valor_libre) para encajar con 'value'."""
        if self.a == value:
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


def generate_all_tiles() -> list:
    return [Tile(a, b) for a in range(7) for b in range(a, 7)]


@dataclass
class GameState:
    board: list = field(default_factory=list)      # fichas colocadas en orden
    left_end: Optional[int] = None                  # extremo izquierdo activo
    right_end: Optional[int] = None                 # extremo derecho activo
    agent_hand: list = field(default_factory=list)
    opponent_hand: list = field(default_factory=list)
    pool: list = field(default_factory=list)
    current_player: int = 0                         # 0=agente, 1=oponente
    pass_count: int = 0                             # pases consecutivos
    history: list = field(default_factory=list)     # historial de jugadas
    agent_passes: list = field(default_factory=list)
    opponent_passes: list = field(default_factory=list)

    @classmethod
    def new_game(cls):
        tiles = generate_all_tiles()
        random.shuffle(tiles)
        agent_hand = tiles[:7]
        opponent_hand = tiles[7:14]
        pool = tiles[14:]

        # Determina quién empieza: el que tenga el doble más alto
        agent_max = max((t for t in agent_hand if t.a == t.b), key=lambda t: t.a, default=None)
        opp_max = max((t for t in opponent_hand if t.a == t.b), key=lambda t: t.a, default=None)

        if agent_max and opp_max:
            first = 0 if agent_max.a >= opp_max.a else 1
        elif agent_max:
            first = 0
        elif opp_max:
            first = 1
        else:
            first = random.randint(0, 1)

        return cls(
            agent_hand=agent_hand,
            opponent_hand=opponent_hand,
            pool=pool,
            current_player=first
        )

    def valid_moves(self, hand: list) -> list:
        """Retorna lista de (ficha, extremo) válidos. extremo: 'left' o 'right'."""
        if self.left_end is None:
            return [(t, 'left') for t in hand]
        moves = []
        for tile in hand:
            if tile.fits(self.left_end):
                moves.append((tile, 'left'))
            if tile.fits(self.right_end) and self.right_end != self.left_end:
                moves.append((tile, 'right'))
            elif tile.fits(self.right_end) and self.right_end == self.left_end:
                if (tile, 'left') not in moves:
                    moves.append((tile, 'right'))
        return moves

    def apply_move(self, tile: Tile, side: str, player: int) -> 'GameState':
        """Retorna un nuevo GameState resultado de aplicar la jugada."""
        import copy
        ns = copy.deepcopy(self)
        hand = ns.agent_hand if player == 0 else ns.opponent_hand

        # Remover ficha de la mano
        hand[:] = [t for t in hand if not (t == tile)]

        if ns.left_end is None:
            ns.board.append(tile)
            ns.left_end = tile.a
            ns.right_end = tile.b
        elif side == 'left':
            _, free = tile.oriented(ns.left_end)
            ns.board.insert(0, tile)
            ns.left_end = free
        else:
            _, free = tile.oriented(ns.right_end)
            ns.board.append(tile)
            ns.right_end = free

        ns.pass_count = 0
        ns.history.append({'player': player, 'tile': tile, 'side': side})
        ns.current_player = 1 - player
        return ns

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
        opp_pips = sum(t.pips() for t in self.opponent_hand)
        if agent_pips < opp_pips:
            return 0
        elif opp_pips < agent_pips:
            return 1
        return -1  # empate

    def pip_sum(self, player: int) -> int:
        hand = self.agent_hand if player == 0 else self.opponent_hand
        return sum(t.pips() for t in hand)