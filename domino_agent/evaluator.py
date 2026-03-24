import math


def manhattan_distance(state, player: int = 0) -> float:
    """
    Distancia Manhattan entre los extremos del tablero y las fichas en mano.
    Mide qué tan 'lejos' en valor están las fichas del jugador respecto al tablero.
    Menor distancia = fichas más conectables.
    """
    if state.left_end is None:
        return 0.0
    hand = state.agent_hand if player == 0 else state.opponent_hand
    if not hand:
        return 0.0
    total = 0.0
    for tile in hand:
        dist_left = min(abs(tile.a - state.left_end), abs(tile.b - state.left_end))
        dist_right = min(abs(tile.a - state.right_end), abs(tile.b - state.right_end))
        total += min(dist_left, dist_right)
    return total / len(hand)


def euclidean_distance(state, player: int = 0) -> float:
    """
    Distancia Euclidiana entre vector de extremos del tablero y pips de fichas en mano.
    Penaliza más fuertemente las fichas muy alejadas en valor.
    """
    if state.left_end is None:
        return 0.0
    hand = state.agent_hand if player == 0 else state.opponent_hand
    if not hand:
        return 0.0
    total = 0.0
    for tile in hand:
        best = min(
            math.sqrt((tile.a - state.left_end)**2 + (tile.b - state.right_end)**2),
            math.sqrt((tile.b - state.left_end)**2 + (tile.a - state.right_end)**2)
        )
        total += best
    return total / len(hand)


def evaluate(state, player: int = 0,
             w1: float = 0.4, w2: float = 0.3,
             w3: float = 0.2, w4: float = 0.1,
             use_manhattan: bool = True,
             use_euclidean: bool = True) -> float:
    """
    f(s) = w1·pip_score + w2·control_ends + w3·block_score + w4·dist_score
    Retorna valor en [-1, 1]. Positivo = favorable para el agente.
    """
    opponent = 1 - player

    # 1. pip_score: diferencia de pips normalizada
    agent_pips = state.pip_sum(player)
    opp_pips = state.pip_sum(opponent)
    max_pips = 6 * 7 * 2  # máximo teórico
    pip_score = (opp_pips - agent_pips) / max_pips

    # 2. control_ends: fichas del agente que encajan en extremos actuales
    if state.left_end is not None:
        agent_hand = state.agent_hand if player == 0 else state.opponent_hand
        opp_hand = state.opponent_hand if player == 0 else state.agent_hand
        agent_fits = sum(1 for t in agent_hand
                         if t.fits(state.left_end) or t.fits(state.right_end))
        opp_fits = sum(1 for t in opp_hand
                       if t.fits(state.left_end) or t.fits(state.right_end))
        control_score = (agent_fits - opp_fits) / max(len(agent_hand) + len(opp_hand), 1)
    else:
        control_score = 0.0

    # 3. block_score: si el oponente tiene pocas jugadas válidas
    opp_moves = state.valid_moves(
        state.opponent_hand if player == 0 else state.agent_hand
    )
    block_score = 1.0 - (len(opp_moves) / 28.0)

    # 4. dist_score: basado en distancias (Manhattan y/o Euclidiana)
    dist_score = 0.0
    divisor = 0
    if use_manhattan:
        m = manhattan_distance(state, player)
        dist_score += -m / 6.0  # normaliza a [-1, 0]
        divisor += 1
    if use_euclidean:
        e = euclidean_distance(state, player)
        dist_score += -e / (6 * math.sqrt(2))
        divisor += 1
    if divisor > 0:
        dist_score /= divisor

    f = w1 * pip_score + w2 * control_score + w3 * block_score + w4 * dist_score
    return max(-1.0, min(1.0, f))