"""
evaluator.py — Funciones heurísticas para la evaluación de estados.

Funciones exportadas:
  manhattan_distance()       — distancia Manhattan mano ↔ extremos del tablero
  euclidean_distance()       — distancia Euclidiana mano ↔ extremos del tablero
  pool_opportunity_score()   — valor esperado de robar del pozo  [0, 1]
  opponent_blocking_score()  — fracción de fichas rivales que encajan
  evaluate()                 — heurística combinada ∈ [-1, 1]

Todas las estrategias importan directamente desde aquí; no duplican lógica.
"""
import math


def manhattan_distance(state, player: int = 0) -> float:
    """
    Distancia Manhattan entre extremos del tablero y fichas en mano.
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
    Distancia Euclidiana entre vector de extremos y pips de fichas en mano.
    Penaliza más fuertemente fichas muy alejadas en valor.
    """
    if state.left_end is None:
        return 0.0
    hand = state.agent_hand if player == 0 else state.opponent_hand
    if not hand:
        return 0.0
    total = 0.0
    for tile in hand:
        best = min(
            math.sqrt((tile.a - state.left_end) ** 2 + (tile.b - state.right_end) ** 2),
            math.sqrt((tile.b - state.left_end) ** 2 + (tile.a - state.right_end) ** 2),
        )
        total += best
    return total / len(hand)


def pool_opportunity_score(state, player: int = 0) -> float:
    """
    Valor esperado de robar del pozo.
    Probabilidad de encontrar al menos una ficha útil. Rango: [0, 1].
    """
    if state.left_end is None or state.pool_size() == 0:
        return 0.0
    expected_left = state.expected_pool_fits(state.left_end)
    expected_right = state.expected_pool_fits(state.right_end)
    best_expected = max(expected_left, expected_right)
    return min(1.0, best_expected / 3.0)


def opponent_blocking_score(state, player: int = 0) -> float:
    """
    Penalización probabilística: fracción de fichas del oponente que encajan.
    """
    if state.left_end is None:
        return 0.0
    opp_hand = state.opponent_hand if player == 0 else state.agent_hand
    if not opp_hand:
        return 1.0
    opp_fits = sum(1 for t in opp_hand
                   if t.fits(state.left_end) or t.fits(state.right_end))
    return opp_fits / max(len(opp_hand), 1)


def evaluate(
    state,
    player: int = 0,
    w1: float = 0.35,
    w2: float = 0.25,
    w3: float = 0.20,
    w4: float = 0.10,
    w5: float = 0.10,
    use_manhattan: bool = True,
    use_euclidean: bool = True,
    use_pool: bool = True,
) -> float:
    """
    Función de evaluación heurística combinada.

    f(s) = w1·pip_score + w2·control_ends + w3·block_score
           + w4·dist_score + w5·pool_score

    Retorna valor en [-1, 1]. Positivo = favorable para el agente.

    Parámetros de distancia:
      use_manhattan  — incluir distancia Manhattan en dist_score
      use_euclidean  — incluir distancia Euclidiana en dist_score
      use_pool       — incluir oportunidad de pozo en el cómputo
    """
    opponent = 1 - player

    # 1. pip_score: diferencia de pips normalizada
    agent_pips = state.pip_sum(player)
    opp_pips = state.pip_sum(opponent)
    max_pips = 6 * 7 * 2
    pip_score = (opp_pips - agent_pips) / max_pips

    # 2. control_ends: fichas del agente que encajan en extremos actuales
    opp_hand = state.opponent_hand if player == 0 else state.agent_hand
    if state.left_end is not None:
        agent_hand = state.agent_hand if player == 0 else state.opponent_hand
        agent_fits = sum(1 for t in agent_hand
                         if t.fits(state.left_end) or t.fits(state.right_end))
        opp_fits = sum(1 for t in opp_hand
                       if t.fits(state.left_end) or t.fits(state.right_end))
        control_score = (agent_fits - opp_fits) / max(len(agent_hand) + len(opp_hand), 1)
    else:
        control_score = 0.0

    # 3. block_score: si el oponente tiene pocas jugadas válidas
    opp_moves = state.valid_moves(opp_hand)
    block_score = 1.0 - (len(opp_moves) / max(len(opp_hand), 1))

    # 4. dist_score: combinación ponderada de distancias habilitadas
    dist_score = 0.0
    divisor = 0
    if use_manhattan:
        dist_score += -manhattan_distance(state, player) / 6.0
        divisor += 1
    if use_euclidean:
        dist_score += -euclidean_distance(state, player) / (6 * math.sqrt(2))
        divisor += 1
    if divisor > 0:
        dist_score /= divisor

    # 5. pool_score: oportunidad de robar ficha útil del pozo
    if use_pool:
        pool_score = pool_opportunity_score(state, player)
        f = (w1 * pip_score + w2 * control_score +
             w3 * block_score + w4 * dist_score +
             w5 * pool_score)
    else:
        total_w = w1 + w2 + w3 + w4
        f = ((w1 / total_w) * pip_score + (w2 / total_w) * control_score +
             (w3 / total_w) * block_score + (w4 / total_w) * dist_score)

    return max(-1.0, min(1.0, f))