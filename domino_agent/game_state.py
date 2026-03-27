import random
from dataclasses import dataclass, field
from typing import Optional, Dict, List


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
    board: list = field(default_factory=list)
    left_end: Optional[int] = None
    right_end: Optional[int] = None
    agent_hand: list = field(default_factory=list)
    opponent_hand: list = field(default_factory=list)
    pool: list = field(default_factory=list)
    current_player: int = 0
    pass_count: int = 0
    history: list = field(default_factory=list)
    agent_passes: list = field(default_factory=list)
    opponent_passes: list = field(default_factory=list)

    # --- Probabilistic knowledge (agent's perspective) ---
    # Conjunto de fichas que el agente NO conoce (oponente + pozo)
    unknown_tiles: list = field(default_factory=list)

    @classmethod
    def new_game(cls):
        tiles = generate_all_tiles()
        random.shuffle(tiles)
        agent_hand = tiles[:7]
        opponent_hand = tiles[7:14]
        pool = tiles[14:]

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

        # Desde la perspectiva del agente, las fichas desconocidas son oponente + pozo
        unknown = list(opponent_hand) + list(pool)

        return cls(
            agent_hand=agent_hand,
            opponent_hand=opponent_hand,
            pool=pool,
            current_player=first,
            unknown_tiles=unknown
        )

    # ------------------------------------------------------------------ #
    #  PROBABILIDADES DESDE LA PERSPECTIVA DEL AGENTE                     #
    # ------------------------------------------------------------------ #

    def pool_size(self) -> int:
        return len(self.pool)

    def opponent_hand_size(self) -> int:
        return len(self.opponent_hand)

    def unknown_size(self) -> int:
        """Total de fichas que el agente no puede ver."""
        return len(self.unknown_tiles)

    def prob_tile_in_pool(self, tile: Tile) -> float:
        """
        Probabilidad de que una ficha específica esté en el pozo,
        dado que el agente sabe que está en el conjunto 'unknown'.
        Usa distribución uniforme sobre las fichas no observadas.
        """
        if tile in self.agent_hand or tile in self.board:
            return 0.0
        total_unknown = self.unknown_size()
        if total_unknown == 0:
            return 0.0
        pool_count = self.pool_size()
        # P(en pozo) = tamaño_pozo / total_desconocidas
        return pool_count / total_unknown

    def prob_tile_in_opponent(self, tile: Tile) -> float:
        """
        Probabilidad de que una ficha esté en la mano del oponente.
        Se refina con las pistas de los pases del oponente.
        """
        if tile in self.agent_hand or tile in self.board:
            return 0.0
        total_unknown = self.unknown_size()
        if total_unknown == 0:
            return 0.0
        opp_size = self.opponent_hand_size()

        # Reducir probabilidad si el oponente pasó en un turno donde la ficha hubiera sido jugable
        penalty = 1.0
        if self.opponent_passes:
            # Si el oponente pasó cuando los extremos coincidían con pips de la ficha,
            # es menos probable que la tenga
            passed_at = set(self.opponent_passes)
            if tile.a in passed_at or tile.b in passed_at:
                penalty = 0.3  # reducción significativa

        base_prob = (opp_size / total_unknown) * penalty
        return min(1.0, base_prob)

    def expected_pool_fits(self, end_value: int) -> float:
        """
        Número esperado de fichas en el pozo que encajan con 'end_value'.
        Útil para decidir si vale la pena pasar para robar del pozo.
        """
        if self.pool_size() == 0:
            return 0.0
        fitting_unknown = sum(
            1 for t in self.unknown_tiles if t.fits(end_value)
        )
        total_unknown = self.unknown_size()
        if total_unknown == 0:
            return 0.0
        # Esperanza: fichas_encajan_desconocidas * (pozo / total_desconocidas)
        return fitting_unknown * (self.pool_size() / total_unknown)

    def draw_from_pool(self, player: int) -> Optional[Tile]:
        """
        El jugador toma una ficha del pozo (si hay).
        Actualiza unknown_tiles en consecuencia.
        """
        if not self.pool:
            return None
        tile = self.pool.pop(0)
        if player == 0:
            self.agent_hand.append(tile)
            # La ficha ya no es desconocida para el agente
            if tile in self.unknown_tiles:
                self.unknown_tiles.remove(tile)
        else:
            self.opponent_hand.append(tile)
            # Sigue siendo desconocida para el agente
        return tile

    # ------------------------------------------------------------------ #
    #  MÉTODOS ORIGINALES                                                  #
    # ------------------------------------------------------------------ #

    def valid_moves(self, hand: list) -> list:
        """Retorna lista de (ficha, extremo) válidos."""
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

    def apply_move(self, tile: 'Tile', side: str, player: int) -> 'GameState':
        import copy
        ns = copy.deepcopy(self)
        hand = ns.agent_hand if player == 0 else ns.opponent_hand

        hand[:] = [t for t in hand if not (t == tile)]

        # La ficha jugada sale de unknown_tiles también
        ns.unknown_tiles = [t for t in ns.unknown_tiles if not (t == tile)]

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

    def apply_draw_and_play(self, player: int) -> 'GameState':
        """
        Implementa la regla correcta del pozo:
        El jugador roba fichas del pozo una a una hasta poder jugar o agotar el pozo.
        Si después de robar puede jugar, retorna el estado con la mejor jugada aplicada.
        Si no puede jugar tras agotar el pozo, retorna None (debe pasar).
        Retorna (nuevo_estado, jugada) o (estado_con_robo, None).
        """
        import copy
        ns = copy.deepcopy(self)
        hand = ns.agent_hand if player == 0 else ns.opponent_hand

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
                return ns, moves  # tiene jugadas disponibles tras robar

        return ns, []  # agotó el pozo sin poder jugar

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
        return -1

    def pip_sum(self, player: int) -> int:
        hand = self.agent_hand if player == 0 else self.opponent_hand
        return sum(t.pips() for t in hand)
