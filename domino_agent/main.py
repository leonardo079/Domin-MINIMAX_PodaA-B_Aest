"""
main.py — Motor de juego con manejo correcto del pozo.

El estado se gestiona de forma centralizada: cuando una estrategia
necesita robar del pozo, actualiza el estado antes de devolver la jugada.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from game_state import GameState
from profiler import CostProfiler
from strategies import STRATEGIES


BANNER = """
╔══════════════════════════════════════════════╗
║        AGENTE DE DOMINÓ — Doble 6            ║
║   A*, Minimax, Manhattan, Euclidiana, Híbrid ║
╚══════════════════════════════════════════════╝
"""

STRATEGY_DESC = {
    'random':    'Jugada aleatoria entre las válidas (baseline)',
    'manhattan': 'Minimax + poda α-β con distancia Manhattan',
    'euclidean': 'Minimax + poda α-β con distancia Euclidiana',
    'astar':     'Búsqueda A* pura (Manhattan + Euclidiana)',
    'hybrid':    'A* + Minimax + α-β + profundidad adaptativa + pozo',
}


def print_state(state: GameState, show_opponent: bool = False):
    print("\n" + "─" * 55)
    if state.board:
        board_str = " ".join(str(t) for t in state.board)
        print(f"  Tablero : {board_str}")
        print(f"  Extremos: [{state.left_end}] ... [{state.right_end}]")
    else:
        print("  Tablero : (vacío)")
    print(f"  Agente  ({len(state.agent_hand)} fichas): "
          + " ".join(str(t) for t in state.agent_hand))
    if show_opponent:
        print(f"  Oponente({len(state.opponent_hand)} fichas): "
              + " ".join(str(t) for t in state.opponent_hand))
    else:
        print(f"  Oponente: {len(state.opponent_hand)} fichas ocultas")
    print(f"  Pozo    : {state.pool_size()} fichas")
    print("─" * 55)


def play_game(strategy_a, strategy_b, verbose=True, show_opp=False):
    """
    Juega una partida. El estado se actualiza centralmente.
    Retorna (ganador, turnos, pips_a, pips_b).
    """
    state = GameState.new_game()
    turn = 0

    while not state.is_terminal():
        turn += 1
        cur = state.current_player

        if verbose:
            print_state(state, show_opp)
            print(f"\n  Turno {turn} — {'Agente' if cur == 0 else 'Oponente'}")

        # ── El jugador decide ──────────────────────────────────────────
        strat = strategy_a if cur == 0 else strategy_b
        hand = state.agent_hand if cur == 0 else state.opponent_hand
        moves = state.valid_moves(hand)

        # Regla del pozo: si no hay jugadas, robar hasta poder jugar
        if not moves and state.pool:
            state, moves = state.apply_draw_and_play(cur)
            if verbose:
                name = 'Agente' if cur == 0 else 'Oponente'
                new_hand = state.agent_hand if cur == 0 else state.opponent_hand
                print(f"  {name} robó del pozo → {len(new_hand)} fichas")

        result = strat.decide(state)

        # ── Aplicar resultado ──────────────────────────────────────────
        if result is None:
            if verbose:
                name = 'Agente' if cur == 0 else 'Oponente'
                print(f"  {name} PASA (sin jugadas, pozo vacío)")
            state = state.apply_pass(cur)
        else:
            tile, side = result
            if verbose:
                name = 'Agente' if cur == 0 else 'Oponente'
                print(f"  {name} juega {tile} → extremo {side}")
            state = state.apply_move(tile, side, cur)

        if verbose and strat.profiler and strat.profiler.metrics:
            m = strat.profiler.metrics[-1]
            print(f"  [{strat.name}] {m.time_ms:.2f}ms | "
                  f"nodos={m.nodes_expanded} | evals={m.eval_calls} | "
                  f"prof={m.max_depth}")

    winner = state.winner()
    if verbose:
        print_state(state, True)
        names = {0: 'AGENTE', 1: 'OPONENTE', -1: 'EMPATE'}
        print(f"\n  ══ FIN — {names.get(winner, '?')} gana ══")
        print(f"  Pips agente: {state.pip_sum(0)} | Pips oponente: {state.pip_sum(1)}")
        print(f"  Turnos: {turn}")

    return winner, turn, state.pip_sum(0), state.pip_sum(1)


def select_strategy(role: str) -> str:
    print(f"\n  Estrategia para {role}:")
    keys = list(STRATEGIES.keys())
    for i, k in enumerate(keys, 1):
        print(f"    [{i}] {k:12s} — {STRATEGY_DESC[k]}")
    while True:
        c = input(f"\n  Opción (1-{len(keys)}): ").strip()
        if c.isdigit() and 1 <= int(c) <= len(keys):
            return keys[int(c) - 1]
        print("  Opción inválida.")


def select_mode() -> str:
    print("\n  Modo de juego:")
    print("    [1] Agente vs Agente (simulación)")
    print("    [2] Agente vs Humano")
    print("    [3] Benchmark rápido (10 partidas)")
    print("    [4] Torneo completo (50 partidas × 5 matchups)")
    while True:
        c = input("\n  Opción (1-4): ").strip()
        if c in ('1', '2', '3', '4'):
            return c
        print("  Opción inválida.")


def main():
    print(BANNER)
    mode = select_mode()

    if mode == '1':
        name_a = select_strategy("Agente A")
        name_b = select_strategy("Agente B")
        prof_a = CostProfiler(name_a)
        prof_b = CostProfiler(name_b)
        sa = STRATEGIES[name_a](player=0)
        sb = STRATEGIES[name_b](player=1)
        sa.set_profiler(prof_a)
        sb.set_profiler(prof_b)
        play_game(sa, sb, verbose=True, show_opp=True)
        os.makedirs('results', exist_ok=True)
        prof_a.export_csv('results/metrics.csv')
        prof_b.export_csv('results/metrics.csv')
        print("\n  Métricas guardadas en results/metrics.csv")

    elif mode == '2':
        name_a = select_strategy("el Agente (IA)")
        prof_a = CostProfiler(name_a)
        sa = STRATEGIES[name_a](player=0)
        sa.set_profiler(prof_a)

        from strategies.base import AgentStrategy
        class HumanStrategy(AgentStrategy):
            @property
            def name(self): return "human"
            def decide(self, state):
                moves = state.valid_moves(state.opponent_hand)
                if not moves:
                    return None
                print("\n  Tus fichas: " + " ".join(
                    f"[{i+1}]{t}" for i, t in enumerate(state.opponent_hand)))
                print("  Jugadas válidas:")
                for i, (t, s) in enumerate(moves, 1):
                    print(f"    [{i}] {t} → {s}")
                while True:
                    c = input("  Elige: ").strip()
                    if c.isdigit() and 1 <= int(c) <= len(moves):
                        return moves[int(c) - 1]
                    print("  Inválido.")

        sb = HumanStrategy(player=1)
        play_game(sa, sb, verbose=True, show_opp=False)

    elif mode in ('3', '4'):
        import subprocess
        n = '10' if mode == '3' else '50'
        print(f"\n  Iniciando benchmark con {n} partidas por matchup...")
        subprocess.run([sys.executable, 'benchmark.py', '--games', n])
        print("\n  ¿Generar gráficas? (requiere matplotlib)")
        if input("  [s/n]: ").strip().lower() == 's':
            subprocess.run([sys.executable, 'plot_results.py'])


if __name__ == '__main__':
    main()
