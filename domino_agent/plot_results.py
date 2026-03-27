"""
plot_results.py — Genera las 4 gráficas equivalentes al paper + gráficas de costo computacional.

Requiere: pip install matplotlib
Uso: python plot_results.py  (después de correr benchmark.py)
"""
import json
import os
import sys

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("Instala matplotlib: pip install matplotlib --break-system-packages")
    sys.exit(1)

RESULTS_DIR = 'results'
COLORS = {
    'a': '#2196F3',   # azul
    'b': '#F44336',   # rojo
    'draw': '#9E9E9E' # gris
}
STYLE = {
    'figure.facecolor': 'white',
    'axes.facecolor': '#F8F9FA',
    'axes.grid': True,
    'grid.alpha': 0.4,
    'font.family': 'DejaVu Sans',
}
plt.rcParams.update(STYLE)


def load_results():
    path = os.path.join(RESULTS_DIR, 'tournament_results.json')
    if not os.path.exists(path):
        print(f"No se encontró {path}. Corre benchmark.py primero.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


# ─── Figura 1: Win Rate por matchup (equivalente al paper) ───────────────────
def plot_win_rates(results):
    labels = [r['label'].replace(' vs ', '\nvs\n') for r in results]
    wins_a = [r['win_rate_a'] for r in results]
    wins_b = [r['win_rate_b'] for r in results]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_a = ax.bar(x - width/2, wins_a, width, color=COLORS['a'],
                    label='Agente 1', alpha=0.85, edgecolor='white')
    bars_b = ax.bar(x + width/2, wins_b, width, color=COLORS['b'],
                    label='Agente 2', alpha=0.85, edgecolor='white')

    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_title('Performance Comparison Across AI Agents — Dominó', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 100)
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.legend(fontsize=11)

    for bar in bars_a:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{bar.get_height():.0f}%', ha='center', va='bottom', fontsize=9)
    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{bar.get_height():.0f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'fig1_win_rates.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig1_win_rates.png")


# ─── Figura 2: Score advantage convergence (equivalente al paper) ────────────
def plot_score_convergence(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_line = ['#1565C0', '#E53935', '#2E7D32', '#F57F17', '#6A1B9A']

    for i, r in enumerate(results):
        adv = r['score_advantage_per_game']
        # Media móvil de 5 partidas
        window = 5
        smoothed = []
        for j in range(len(adv)):
            start = max(0, j - window + 1)
            smoothed.append(sum(adv[start:j+1]) / (j - start + 1))

        ax.plot(range(1, len(smoothed)+1), smoothed,
                label=r['label'], color=colors_line[i % len(colors_line)],
                linewidth=2, alpha=0.85)

    ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
    ax.set_xlabel('Número de partida', fontsize=12)
    ax.set_ylabel('Ventaja en pips (P1 - P2)', fontsize=12)
    ax.set_title('Score Advantage Convergence per Game', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'fig2_score_convergence.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig2_score_convergence.png")


# ─── Figura 3: Manhattan vs Euclidean detalle ────────────────────────────────
def plot_manhattan_vs_euclidean(results):
    # Buscar matchup de distancias
    dist_r = next((r for r in results if r['tag'] == 'dist_cmp'), None)
    if not dist_r:
        return

    labels = ['Manhattan\nvs Random', 'Manhattan vs Euclidean']
    # Manhattan vs Random: buscar ese matchup
    astar_r = next((r for r in results if r['tag'] == 'astar_m'), None)
    wins1 = astar_r['wins_a'] if astar_r else 0
    wins2 = dist_r['wins_a']  # Manhattan gana vs Euclidean

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, [wins1, wins2], color=[COLORS['a'], COLORS['b']],
                  width=0.5, alpha=0.85, edgecolor='white')
    ax.set_ylabel('Victorias (de 50)', fontsize=12)
    ax.set_title('Heuristic Comparison: Manhattan vs Euclidean', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 50)

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(int(bar.get_height())), ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'fig3_manhattan_vs_euclidean.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig3_manhattan_vs_euclidean.png")


# ─── Figura 4: Distribución de turnos (equivalente al paper) ─────────────────
def plot_turn_distribution(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_line = ['#1565C0', '#E53935', '#2E7D32', '#F57F17', '#6A1B9A']

    all_turns = []
    for r in results:
        all_turns.extend(r['turns_per_game'])
    min_t, max_t = min(all_turns), max(all_turns)
    bins = range(min_t, max_t + 2)

    for i, r in enumerate(results):
        ax.hist(r['turns_per_game'], bins=bins, alpha=0.5,
                label=r['label'], color=colors_line[i % len(colors_line)],
                edgecolor='white')

    ax.set_xlabel('Número de turnos', fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)
    ax.set_title('Distribution of Game Length', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'fig4_turn_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig4_turn_distribution.png")


# ─── Figura 5 (NUEVA): Costo computacional por estrategia ────────────────────
def plot_computational_cost(results):
    """
    Gráfica que el paper NO tiene.
    Compara nodos, tiempo y evals entre estrategias.
    """
    # Recopilar métricas únicas por estrategia
    strategy_metrics = {}
    for r in results:
        for key, side in [('metrics_a', 'agent_a'), ('metrics_b', 'agent_b')]:
            name = r[side]
            m = r[key]
            if m and name not in strategy_metrics:
                strategy_metrics[name] = m

    strategies = list(strategy_metrics.keys())
    avg_nodes = [strategy_metrics[s].get('avg_nodes', 0) for s in strategies]
    avg_time = [strategy_metrics[s].get('avg_time_ms', 0) for s in strategies]
    avg_evals = [strategy_metrics[s].get('avg_evals', 0) for s in strategies]
    avg_depth = [strategy_metrics[s].get('avg_depth', 0) for s in strategies]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Computational Cost Analysis — Métricas no reportadas en el paper',
                 fontsize=14, fontweight='bold')

    palette = ['#1565C0', '#E53935', '#2E7D32', '#F57F17', '#6A1B9A']
    colors = [palette[i % len(palette)] for i in range(len(strategies))]

    # Nodos promedio por turno
    axes[0, 0].bar(strategies, avg_nodes, color=colors, alpha=0.85, edgecolor='white')
    axes[0, 0].set_title('Avg Nodes Expanded / Turn', fontweight='bold')
    axes[0, 0].set_ylabel('Nodos')
    for i, v in enumerate(avg_nodes):
        axes[0, 0].text(i, v + max(avg_nodes)*0.01, f'{v:.1f}', ha='center', fontsize=9)

    # Tiempo promedio por turno
    axes[0, 1].bar(strategies, avg_time, color=colors, alpha=0.85, edgecolor='white')
    axes[0, 1].set_title('Avg Time per Turn (ms)', fontweight='bold')
    axes[0, 1].set_ylabel('Tiempo (ms)')
    for i, v in enumerate(avg_time):
        axes[0, 1].text(i, v + max(avg_time)*0.01, f'{v:.2f}', ha='center', fontsize=9)

    # Evaluaciones heurísticas por turno
    axes[1, 0].bar(strategies, avg_evals, color=colors, alpha=0.85, edgecolor='white')
    axes[1, 0].set_title('Avg Heuristic Evaluations / Turn', fontweight='bold')
    axes[1, 0].set_ylabel('Llamadas a evaluate()')
    for i, v in enumerate(avg_evals):
        axes[1, 0].text(i, v + max(avg_evals)*0.01 if max(avg_evals) > 0 else 0.1,
                        f'{v:.1f}', ha='center', fontsize=9)

    # Profundidad promedio
    axes[1, 1].bar(strategies, avg_depth, color=colors, alpha=0.85, edgecolor='white')
    axes[1, 1].set_title('Avg Search Depth Reached', fontweight='bold')
    axes[1, 1].set_ylabel('Profundidad')
    axes[1, 1].axhline(4, color='gray', linestyle='--', alpha=0.6, label='DEPTH=4 base')
    axes[1, 1].legend(fontsize=9)
    for i, v in enumerate(avg_depth):
        axes[1, 1].text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=9)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'fig5_computational_cost.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig5_computational_cost.png")


# ─── Figura 6 (NUEVA): Costo vs Rendimiento ──────────────────────────────────
def plot_cost_vs_performance(results):
    """
    Scatter plot: nodos_promedio vs win_rate.
    Responde la pregunta clave: ¿más costo = mejores resultados?
    """
    strategy_metrics = {}
    for r in results:
        for key, side in [('metrics_a', 'agent_a'), ('metrics_b', 'agent_b')]:
            name = r[side]
            m = r[key]
            if m and name not in strategy_metrics:
                strategy_metrics[name] = m

    # Win rates contra Random como referencia común
    win_vs_random = {}
    for r in results:
        if r['agent_b'] == 'random':
            win_vs_random[r['agent_a']] = r['win_rate_a']
        elif r['agent_a'] == 'random':
            win_vs_random[r['agent_b']] = r['win_rate_b']

    fig, ax = plt.subplots(figsize=(9, 6))
    palette = ['#1565C0', '#E53935', '#2E7D32', '#F57F17', '#6A1B9A']

    plotted = []
    for i, (strat, metrics) in enumerate(strategy_metrics.items()):
        if strat not in win_vs_random or strat == 'random':
            continue
        nodes = metrics.get('avg_nodes', 0)
        wr = win_vs_random[strat]
        ax.scatter(nodes, wr, s=200, color=palette[i % len(palette)],
                   zorder=5, edgecolors='white', linewidths=1.5)
        ax.annotate(strat, (nodes, wr),
                    textcoords="offset points", xytext=(8, 4), fontsize=10)
        plotted.append((nodes, wr, strat))

    if plotted:
        xs = [p[0] for p in plotted]
        ys = [p[1] for p in plotted]
        if len(xs) > 1:
            z = np.polyfit(xs, ys, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(xs), max(xs), 100)
            ax.plot(x_line, p(x_line), '--', color='gray', alpha=0.5, label='Tendencia')

    ax.axhline(50, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('Nodos promedio por turno (costo computacional)', fontsize=12)
    ax.set_ylabel('Win Rate vs Random (%)', fontsize=12)
    ax.set_title('Cost vs Performance: ¿Más cómputo = Mejores decisiones?',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'fig6_cost_vs_performance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig6_cost_vs_performance.png")


def main():
    print("\n  Generando gráficas...")
    results = load_results()

    plot_win_rates(results)
    plot_score_convergence(results)
    plot_manhattan_vs_euclidean(results)
    plot_turn_distribution(results)
    plot_computational_cost(results)
    plot_cost_vs_performance(results)

    print(f"\n  6 gráficas guardadas en {RESULTS_DIR}/")
    print("  Figs 1-4: equivalentes al paper")
    print("  Figs 5-6: métricas de costo computacional (contribución diferencial)")


if __name__ == '__main__':
    main()
