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


def pool_opportunity_score(state, player: int = 0) -> float:
    """
    NUEVA MÉTRICA: Valor esperado de robar del pozo.

    Calcula la probabilidad de que al menos una ficha del pozo
    sea jugable en los extremos actuales del tablero.
    Rango: [0, 1]. 1 = muy probable encontrar ficha útil en el pozo.
    """
    if state.left_end is None or state.pool_size() == 0:
        return 0.0

    expected_left = state.expected_pool_fits(state.left_end)
    expected_right = state.expected_pool_fits(state.right_end)

    # Probabilidad de encontrar al menos una ficha útil (aproximación)
    best_expected = max(expected_left, expected_right)
    # Normalizar: máximo teórico ~7 fichas por valor
    return min(1.0, best_expected / 3.0)


def opponent_blocking_score(state, player: int = 0) -> float:
    """
    NUEVA MÉTRICA: Penalización probabilística basada en fichas que el oponente
    probablemente tiene y pueden bloquear al agente.

    Usa las probabilidades del pozo para inferir fuerza del oponente.
    """
    if state.left_end is None:
        return 0.0

    opponent = 1 - player
    opp_hand = state.opponent_hand if player == 0 else state.agent_hand

    if not opp_hand:
        return 1.0

    # Fichas del oponente que encajan en extremos actuales
    opp_fits = sum(1 for t in opp_hand
                   if t.fits(state.left_end) or t.fits(state.right_end))

    # Score: cuántas fichas del oponente son jugables
    return opp_fits / max(len(opp_hand), 1)


def evaluate(state, player: int = 0,
             w1: float = 0.35, w2: float = 0.25,
             w3: float = 0.20, w4: float = 0.10,
             w5: float = 0.10,
             use_manhattan: bool = True,
             use_euclidean: bool = True,
             use_pool: bool = True) -> float:
    """
    f(s) = w1·pip_score + w2·control_ends + w3·block_score
           + w4·dist_score + w5·pool_score

    Retorna valor en [-1, 1]. Positivo = favorable para el agente.

    Parámetros:
    -----------
    use_pool : bool
        Si True, incluye la métrica de oportunidad del pozo (w5).
        Permite considerar las 28 fichas en la evaluación.
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
        dist_score += -m / 6.0
        divisor += 1
    if use_euclidean:
        e = euclidean_distance(state, player)
        dist_score += -e / (6 * math.sqrt(2))
        divisor += 1
    if divisor > 0:
        dist_score /= divisor

    # 5. pool_score: oportunidad de robar ficha útil del pozo (NUEVO)
    pool_score = 0.0
    if use_pool:
        pool_score = pool_opportunity_score(state, player)
        # Reajustar pesos para incluir w5
        f = (w1 * pip_score + w2 * control_score +
             w3 * block_score + w4 * dist_score +
             w5 * pool_score)
    else:
        # Sin pool: redistribuir peso w5 entre los demás
        total_w = w1 + w2 + w3 + w4
        f = ((w1/total_w) * pip_score + (w2/total_w) * control_score +
             (w3/total_w) * block_score + (w4/total_w) * dist_score)

    return max(-1.0, min(1.0, f))
