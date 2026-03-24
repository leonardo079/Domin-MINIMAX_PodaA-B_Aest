import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from game_state import GameState
from profiler import CostProfiler
from strategies import STRATEGIES


BANNER = """
╔══════════════════════════════════════════════╗
║        AGENTE DE DOMINÓ — Doble 6            ║
║   IA con A*, Minimax, Manhattan, Euclidiana  ║
╚══════════════════════════════════════════════╝
"""

STRATEGY_DESC = {
    'random':    'Jugada aleatoria entre las válidas (baseline)',
    'manhattan': 'Minimax + poda α-β con distancia Manhattan',
    'euclidean': 'Minimax + poda α-β con distancia Euclidiana',
    'astar':     'Búsqueda A* pura (Manhattan + Euclidiana)',
    'hybrid':    'A* + Minimax + α-β con ambas distancias (modelo completo)',
}


def print_state(state: GameState, show_opponent: bool = False):
    print("\n" + "─" * 50)
    if state.board:
        board_str = " ".join(str(t) for t in state.board)
        print(f"  Tablero: {board_str}")
        print(f"  Extremos: [{state.left_end}] ... [{state.right_end}]")
    else:
        print("  Tablero: (vacío)")
    print(f"  Mano agente ({len(state.agent_hand)} fichas): "
          + " ".join(str(t) for t in state.agent_hand))
    if show_opponent:
        print(f"  Mano oponente ({len(state.opponent_hand)} fichas): "
              + " ".join(str(t) for t in state.opponent_hand))
    else:
        print(f"  Mano oponente: {len(state.opponent_hand)} fichas ocultas")
    print("─" * 50)


def select_strategy(role: str) -> str:
    print(f"\n  Selecciona estrategia para {role}:")
    keys = list(STRATEGIES.keys())
    for i, k in enumerate(keys, 1):
        print(f"    [{i}] {k:12s} — {STRATEGY_DESC[k]}")
    while True:
        choice = input(f"\n  Opción (1-{len(keys)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(keys):
            return keys[int(choice) - 1]
        print("  Opción inválida, intenta de nuevo.")


def select_mode() -> str:
    print("\n  Modo de juego:")
    print("    [1] Agente vs Agente (simulación automática)")
    print("    [2] Agente vs Humano")
    print("    [3] Benchmark (N partidas automáticas + métricas)")
    while True:
        c = input("\n  Opción (1-3): ").strip()
        if c in ('1', '2', '3'):
            return c
        print("  Opción inválida.")


def play_game(strategy_a, strategy_b, verbose=True, show_opp=False):
    """
    Juega una partida completa entre strategy_a (agente) y strategy_b (oponente).
    Retorna: (ganador: 0|1|-1, turnos: int)
    """
    state = GameState.new_game()
    turn = 0

    while not state.is_terminal():
        turn += 1
        if verbose:
            print_state(state, show_opp)
            print(f"\n  Turno {turn} — Jugador: {'Agente' if state.current_player == 0 else 'Oponente'}")

        if state.current_player == 0:
            result = strategy_a.decide(state)
            if verbose and strategy_a.profiler and strategy_a.profiler.metrics:
                m = strategy_a.profiler.metrics[-1]
                print(f"  [{strategy_a.name}] Tiempo: {m.time_ms:.2f}ms | "
                      f"Nodos: {m.nodes_expanded} | Evals: {m.eval_calls} | "
                      f"Prof: {m.max_depth}")
        else:
            result = strategy_b.decide(state)
            if verbose and strategy_b.profiler and strategy_b.profiler.metrics:
                m = strategy_b.profiler.metrics[-1]
                print(f"  [{strategy_b.name}] Tiempo: {m.time_ms:.2f}ms | "
                      f"Nodos: {m.nodes_expanded} | Evals: {m.eval_calls} | "
                      f"Prof: {m.max_depth}")

        if result is None:
            if verbose:
                name = 'Agente' if state.current_player == 0 else 'Oponente'
                print(f"  {name} pasa (sin jugadas válidas)")
            state = state.apply_pass(state.current_player)
        else:
            tile, side = result
            if verbose:
                name = 'Agente' if state.current_player == 0 else 'Oponente'
                print(f"  {name} juega {tile} en extremo {side}")
            state = state.apply_move(tile, side, state.current_player)

    winner = state.winner()
    if verbose:
        print_state(state, True)
        names = {0: 'Agente', 1: 'Oponente', -1: 'Empate'}
        print(f"\n  ══ FIN DE PARTIDA — Ganador: {names.get(winner, '?')} ══")
        print(f"  Pips agente: {state.pip_sum(0)} | Pips oponente: {state.pip_sum(1)}")
        print(f"  Turnos jugados: {turn}")

    return winner, turn


def benchmark(strategy_name_a: str, strategy_name_b: str, n: int = 20):
    print(f"\n  Benchmark: {strategy_name_a} vs {strategy_name_b} — {n} partidas\n")
    wins = {0: 0, 1: 0, -1: 0}
    total_turns = 0

    prof_a = CostProfiler(strategy_name_a)
    prof_b = CostProfiler(strategy_name_b)

    for i in range(n):
        sa = STRATEGIES[strategy_name_a](player=0)
        sb = STRATEGIES[strategy_name_b](player=1)
        sa.set_profiler(prof_a)
        sb.set_profiler(prof_b)
        winner, turns = play_game(sa, sb, verbose=False)
        wins[winner] = wins.get(winner, 0) + 1
        total_turns += turns
        print(f"  Partida {i+1:3d}/{n} — Ganador: {'A' if winner==0 else 'B' if winner==1 else 'Empate'} | Turnos: {turns}")

    print("\n  ── Resultados ─────────────────────────────")
    print(f"  Victoria {strategy_name_a}: {wins[0]}/{n} ({wins[0]/n*100:.1f}%)")
    print(f"  Victoria {strategy_name_b}: {wins[1]}/{n} ({wins[1]/n*100:.1f}%)")
    print(f"  Empates: {wins[-1]}/{n}")
    print(f"  Promedio de turnos: {total_turns/n:.1f}")

    print("\n  ── Métricas computacionales ───────────────")
    for name, prof in [(strategy_name_a, prof_a), (strategy_name_b, prof_b)]:
        s = prof.summary()
        if s:
            print(f"\n  [{name}]")
            print(f"    Tiempo promedio/turno : {s['avg_time_ms']:.3f} ms")
            print(f"    Tiempo máximo/turno   : {s['max_time_ms']:.3f} ms")
            print(f"    Tiempo total          : {s['total_time_ms']:.1f} ms")
            print(f"    Nodos promedio/turno  : {s['avg_nodes']:.1f}")
            print(f"    Nodos totales         : {s['total_nodes']}")
            print(f"    Evals promedio/turno  : {s['avg_evals']:.1f}")
            print(f"    Profundidad promedio  : {s['avg_depth']:.1f}")

    prof_a.export_csv('results/metrics.csv')
    prof_b.export_csv('results/metrics.csv')
    print("\n  Métricas exportadas a results/metrics.csv")


def main():
    print(BANNER)
    mode = select_mode()

    if mode == '1':
        print("\n  ── Configuración: Agente A ──")
        name_a = select_strategy("Agente A")
        print("\n  ── Configuración: Agente B ──")
        name_b = select_strategy("Agente B")

        prof_a = CostProfiler(name_a)
        prof_b = CostProfiler(name_b)

        sa = STRATEGIES[name_a](player=0)
        sb = STRATEGIES[name_b](player=1)
        sa.set_profiler(prof_a)
        sb.set_profiler(prof_b)

        play_game(sa, sb, verbose=True, show_opp=True)

        prof_a.export_csv('results/metrics.csv')
        prof_b.export_csv('results/metrics.csv')
        print("\n  Métricas guardadas en results/metrics.csv")

    elif mode == '2':
        print("\n  ── Configuración: Agente ──")
        name_a = select_strategy("el Agente (IA)")

        prof_a = CostProfiler(name_a)
        sa = STRATEGIES[name_a](player=0)
        sa.set_profiler(prof_a)

        # Oponente humano: estrategia especial
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
                    print(f"    [{i}] {t} en extremo {s}")
                while True:
                    c = input("  Elige (número): ").strip()
                    if c.isdigit() and 1 <= int(c) <= len(moves):
                        return moves[int(c) - 1]
                    print("  Opción inválida.")

        sb = HumanStrategy(player=1)
        play_game(sa, sb, verbose=True, show_opp=False)
        prof_a.export_csv('results/metrics.csv')

    elif mode == '3':
        print("\n  ── Benchmark ──")
        print("\n  Estrategia A:")
        name_a = select_strategy("Agente A")
        print("\n  Estrategia B:")
        name_b = select_strategy("Agente B")
        n_str = input("\n  Número de partidas (default 20): ").strip()
        n = int(n_str) if n_str.isdigit() and int(n_str) > 0 else 20
        benchmark(name_a, name_b, n)


if __name__ == '__main__':
    main()