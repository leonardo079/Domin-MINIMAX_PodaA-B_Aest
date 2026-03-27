"""
benchmark.py — Torneo completo entre estrategias.
5 matchups × 50 partidas = 250 total (igual que el paper).
Exporta resultados y métricas de costo computacional.

Uso:
    python benchmark.py              # 50 partidas por matchup
    python benchmark.py --games 10   # modo rápido
"""
import sys, os, json, csv, argparse, time
sys.path.insert(0, os.path.dirname(__file__))

from game_state import GameState
from profiler import CostProfiler
from strategies import STRATEGIES

TOURNAMENT = [
    ('minimax_m', 'manhattan', 'random',    'Minimax(Manhattan) vs Random'),
    ('astar_m',   'astar',     'random',    'A*(Manhattan) vs Random'),
    ('dist_cmp',  'manhattan', 'euclidean', 'Manhattan vs Euclidean'),
    ('hybrid_mm', 'hybrid',    'manhattan', 'Hybrid vs Minimax'),
    ('hybrid_r',  'hybrid',    'random',    'Hybrid vs Random'),
]


def play_game(strategy_a, strategy_b):
    state = GameState.new_game()
    turn = 0
    while not state.is_terminal():
        turn += 1
        cur = state.current_player
        hand = state.agent_hand if cur == 0 else state.opponent_hand
        moves = state.valid_moves(hand)

        if not moves and state.pool:
            state, moves = state.apply_draw_and_play(cur)

        strat = strategy_a if cur == 0 else strategy_b
        result = strat.decide(state)

        if result is None:
            state = state.apply_pass(cur)
        else:
            tile, side = result
            state = state.apply_move(tile, side, cur)

    return state.winner(), turn, state.pip_sum(0), state.pip_sum(1)


def run_matchup(tag, name_a, name_b, label, n_games, results_dir):
    print(f"\n  ▶ {label} ({n_games} partidas)", flush=True)

    wins = {0: 0, 1: 0, -1: 0}
    turns_list, score_advantage = [], []

    # Profilers persistentes para agregar métricas de todas las partidas
    prof_a = CostProfiler(name_a)
    prof_b = CostProfiler(name_b)

    t_start = time.time()
    for i in range(n_games):
        sa = STRATEGIES[name_a](player=0)
        sb = STRATEGIES[name_b](player=1)
        sa.set_profiler(prof_a)
        sb.set_profiler(prof_b)

        winner, turns, pa, pb = play_game(sa, sb)
        wins[winner] = wins.get(winner, 0) + 1
        turns_list.append(turns)
        score_advantage.append(pb - pa)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"    {i+1}/{n_games} — {elapsed:.1f}s", flush=True)

    ma = prof_a.summary()
    mb = prof_b.summary()

    result = {
        'tag': tag, 'label': label,
        'agent_a': name_a, 'agent_b': name_b,
        'n_games': n_games,
        'wins_a': wins[0], 'wins_b': wins[1], 'draws': wins[-1],
        'win_rate_a': round(wins[0] / n_games * 100, 1),
        'win_rate_b': round(wins[1] / n_games * 100, 1),
        'avg_turns': round(sum(turns_list) / len(turns_list), 2),
        'score_advantage_per_game': score_advantage,
        'turns_per_game': turns_list,
        'metrics_a': ma, 'metrics_b': mb,
    }

    prof_a.export_csv(os.path.join(results_dir, f'metrics_{tag}_a.csv'))
    prof_b.export_csv(os.path.join(results_dir, f'metrics_{tag}_b.csv'))

    print(f"    {name_a}: {wins[0]}W  |  {name_b}: {wins[1]}W  |  Empates: {wins[-1]}")
    print(f"    Turnos promedio: {result['avg_turns']}")
    for nm, s in [(name_a, ma), (name_b, mb)]:
        if s:
            print(f"    [{nm:12s}] t̄={s['avg_time_ms']}ms  "
                  f"nodos̄={s['avg_nodes']}  evals̄={s['avg_evals']}  prof̄={s['avg_depth']}")

    return result


def run_tournament(n_games=50, results_dir='results'):
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n{'═'*60}")
    print(f"  TORNEO — {len(TOURNAMENT)} matchups × {n_games} partidas = {len(TOURNAMENT)*n_games} total")
    print(f"{'═'*60}")

    all_results = []
    t0 = time.time()
    for tag, na, nb, label in TOURNAMENT:
        r = run_matchup(tag, na, nb, label, n_games, results_dir)
        all_results.append(r)

    # Guardar JSON completo
    with open(os.path.join(results_dir, 'tournament_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # CSV resumen (equivalente a Table I del paper)
    with open(os.path.join(results_dir, 'summary_table.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Matchup','A1_Wins','A2_Wins','Draws','WinRate_A1%','WinRate_A2%',
                         'Avg_Turns','A1_AvgTime_ms','A1_AvgNodes','A1_AvgEvals','A1_AvgDepth',
                         'A2_AvgTime_ms','A2_AvgNodes','A2_AvgEvals','A2_AvgDepth'])
        for r in all_results:
            ma, mb = r['metrics_a'], r['metrics_b']
            writer.writerow([
                r['label'], r['wins_a'], r['wins_b'], r['draws'],
                r['win_rate_a'], r['win_rate_b'], r['avg_turns'],
                ma.get('avg_time_ms',0), ma.get('avg_nodes',0),
                ma.get('avg_evals',0),   ma.get('avg_depth',0),
                mb.get('avg_time_ms',0), mb.get('avg_nodes',0),
                mb.get('avg_evals',0),   mb.get('avg_depth',0),
            ])

    total = time.time() - t0
    print(f"\n{'═'*60}")
    print(f"  TABLA RESUMEN")
    print(f"{'═'*60}")
    print(f"  {'Matchup':<38} {'A1W':>4} {'A2W':>4} {'Emp':>4} {'Turn̄':>6}")
    print(f"  {'─'*58}")
    for r in all_results:
        print(f"  {r['label']:<38} {r['wins_a']:>4} {r['wins_b']:>4} "
              f"{r['draws']:>4} {r['avg_turns']:>6.1f}")
    print(f"\n  Tiempo total: {total:.1f}s")
    print(f"  Resultados en: {results_dir}/")
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=50)
    args = parser.parse_args()
    run_tournament(n_games=args.games)
