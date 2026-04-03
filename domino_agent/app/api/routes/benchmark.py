"""
routes/benchmark.py — Endpoints de benchmark/torneo entre estrategias.

Genera:
  - CSVs optimizados para las gráficas principales
  - Gráficas PNG a partir de esos CSVs
"""
import asyncio
import csv
import json
import math
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.schemas import BenchmarkRequest
from app.core.profiler import CostProfiler
from app.core.game_runner import play_full_game
from app.strategies import STRATEGIES

router = APIRouter()

RESULTS_DIR = Path("benchmark_results")
PLOTS_DIRNAME = "plots"

DEFAULT_TOURNAMENT = [
    ("minimax_m", "manhattan", "random", "Minimax(Manhattan) vs Random"),
    ("astar_m", "astar", "random", "A*(Manhattan) vs Random"),
    ("dist_cmp", "manhattan", "euclidean", "Manhattan vs Euclidean"),
    ("hybrid_mm", "hybrid", "manhattan", "Hybrid vs Minimax(Manhattan)"),
    ("hybrid_r", "hybrid", "random", "Hybrid vs Random"),
    ("minimax_e", "euclidean", "random", "Minimax(Euclidean) vs Random"),
    ("hybrid_ast", "hybrid", "astar", "Hybrid vs A*(Manhattan)"),
]

SUMMARY_METRIC_KEYS = [
    "avg_time_ms",
    "avg_nodes",
    "avg_evals",
    "avg_depth",
    "turns",
]

RAW_METRIC_KEYS = [
    "time_ms",
    "nodes_expanded",
    "eval_calls",
    "max_depth",
]

PLOT_EXPORT_FORMATS = ("png",)


def _safe_filename(label: str) -> str:
    return (
        label.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("*", "star")
        .replace("/", "_")
    )


def _round(value, digits=4):
    if isinstance(value, (int, float)):
        return round(value, digits)
    return value


def _metric_value(metric, key, default=0):
    if isinstance(metric, dict):
        return metric.get(key, default)
    return getattr(metric, key, default)


def _cumulative_rows(metrics, metric_key: str) -> list[dict]:
    total = 0.0
    rows = []
    for metric in metrics:
        total += float(_metric_value(metric, metric_key, 0) or 0)
        rows.append(
            {
                "turn": _metric_value(metric, "turn", 0),
                "metric": metric_key,
                "cumulative_value": round(total, 4),
            }
        )
    return rows


def _depth_histogram(metrics) -> dict[int, int]:
    counts = Counter(int(_metric_value(m, "max_depth", 0) or 0) for m in metrics)
    return dict(sorted(counts.items()))


def _summary_for_agent(agent_name: str, role: str, summary: dict, pip_values: list[int]) -> dict:
    return {
        "agent": agent_name,
        "role": role,
        "avg_time_ms": _round(summary.get("avg_time_ms", 0), 4),
        "avg_nodes": _round(summary.get("avg_nodes", 0), 4),
        "avg_evals": _round(summary.get("avg_evals", 0), 4),
        "avg_depth": _round(summary.get("avg_depth", 0), 4),
        "total_turns": int(summary.get("turns", 0) or 0),
        "avg_pip_sum": _round(sum(pip_values) / len(pip_values), 4) if pip_values else 0.0,
        "min_pip_sum": min(pip_values) if pip_values else 0,
        "max_pip_sum": max(pip_values) if pip_values else 0,
    }


def _run_matchup(tag: str, name_a: str, name_b: str, label: str, n_games: int) -> dict:
    wins = {0: 0, 1: 0, -1: 0}
    turns_list: list[int] = []
    score_advantage: list[int] = []
    pip_sum_a_per_game: list[int] = []
    pip_sum_b_per_game: list[int] = []

    combined_prof_a = CostProfiler(name_a)
    combined_prof_b = CostProfiler(name_b)

    raw_turn_metrics: list[dict] = []
    cumulative_metrics: list[dict] = []
    game_results: list[dict] = []

    for game_index in range(1, n_games + 1):
        prof_a = CostProfiler(name_a)
        prof_b = CostProfiler(name_b)

        sa = STRATEGIES[name_a](player=0)
        sb = STRATEGIES[name_b](player=1)
        sa.set_profiler(prof_a)
        sb.set_profiler(prof_b)

        winner, turns, pa, pb = play_full_game(sa, sb, prof_a, prof_b)

        wins[winner] = wins.get(winner, 0) + 1
        turns_list.append(turns)
        score_advantage.append(pb - pa)
        pip_sum_a_per_game.append(pa)
        pip_sum_b_per_game.append(pb)

        combined_prof_a.metrics.extend(prof_a.metrics)
        combined_prof_b.metrics.extend(prof_b.metrics)

        game_results.append(
            {
                "game_index": game_index,
                "winner": winner,
                "turns": turns,
                "pip_sum_a": pa,
                "pip_sum_b": pb,
                "score_advantage": pb - pa,
            }
        )

        for role, agent_name, metrics in [
            ("A", name_a, prof_a.metrics),
            ("B", name_b, prof_b.metrics),
        ]:
            for metric in metrics:
                raw_turn_metrics.append(
                    {
                        "matchup": label,
                        "tag": tag,
                        "game_index": game_index,
                        "agent": agent_name,
                        "role": role,
                        "turn": _metric_value(metric, "turn", 0),
                        "time_ms": _round(_metric_value(metric, "time_ms", 0), 4),
                        "nodes_expanded": _metric_value(metric, "nodes_expanded", 0),
                        "eval_calls": _metric_value(metric, "eval_calls", 0),
                        "max_depth": _metric_value(metric, "max_depth", 0),
                    }
                )

            for metric_key in RAW_METRIC_KEYS[:3]:
                for row in _cumulative_rows(metrics, metric_key):
                    cumulative_metrics.append(
                        {
                            "matchup": label,
                            "tag": tag,
                            "game_index": game_index,
                            "agent": agent_name,
                            "role": role,
                            **row,
                        }
                    )

    metrics_a_summary = combined_prof_a.summary()
    metrics_b_summary = combined_prof_b.summary()

    summary_rows = [
        _summary_for_agent(name_a, "A", metrics_a_summary, pip_sum_a_per_game),
        _summary_for_agent(name_b, "B", metrics_b_summary, pip_sum_b_per_game),
    ]

    return {
        "tag": tag,
        "label": label,
        "agent_a": name_a,
        "agent_b": name_b,
        "n_games": n_games,
        "wins_a": wins[0],
        "wins_b": wins[1],
        "draws": wins[-1],
        "win_rate_a": round(wins[0] / n_games * 100, 1),
        "win_rate_b": round(wins[1] / n_games * 100, 1),
        "avg_turns": round(sum(turns_list) / len(turns_list), 2),
        "turns_per_game": turns_list,
        "score_advantage_per_game": score_advantage,
        "pip_sum_a_per_game": pip_sum_a_per_game,
        "pip_sum_b_per_game": pip_sum_b_per_game,
        "game_results": game_results,
        "metrics_a": metrics_a_summary,
        "metrics_b": metrics_b_summary,
        "summary_rows": summary_rows,
        "raw_turn_metrics": raw_turn_metrics,
        "cumulative_metrics": cumulative_metrics,
        "depth_hist_a": _depth_histogram(combined_prof_a.metrics),
        "depth_hist_b": _depth_histogram(combined_prof_b.metrics),
    }


def _export_csv(path: Path, header: list[str], rows: list[list]) -> str:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return str(path)


def _read_csv_dicts(path: str) -> list[dict]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _maybe_float(value):
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save_current_figure(plt, path_no_ext: Path) -> list[str]:
    saved = []
    for ext in PLOT_EXPORT_FORMATS:
        out = path_no_ext.with_suffix(f".{ext}")
        plt.savefig(out, dpi=160, bbox_inches="tight")
        saved.append(str(out))
    plt.close()
    return saved


def _normalize_series(values: list[float]) -> list[float]:
    if not values:
        return []
    max_v = max(values)
    if max_v == 0:
        return [0.0 for _ in values]
    return [v / max_v for v in values]


def _build_model_summary(results: list[dict]) -> list[dict]:
    model_stats = defaultdict(
        lambda: {
            "win_rate": [],
            "avg_time_ms": [],
            "avg_nodes": [],
            "avg_evals": [],
            "avg_depth": [],
            "total_turns": [],
            "avg_pip_sum": [],
        }
    )

    for r in results:
        for role, key, win_key in [
            ("A", "metrics_a", "win_rate_a"),
            ("B", "metrics_b", "win_rate_b"),
        ]:
            agent = r["agent_a"] if role == "A" else r["agent_b"]
            metrics = r.get(key, {}) or {}
            summary_row = next(
                (
                    x
                    for x in r.get("summary_rows", [])
                    if x["agent"] == agent and x["role"] == role
                ),
                None,
            )

            model_stats[agent]["win_rate"].append(float(r.get(win_key, 0) or 0))
            model_stats[agent]["avg_time_ms"].append(float(metrics.get("avg_time_ms", 0) or 0))
            model_stats[agent]["avg_nodes"].append(float(metrics.get("avg_nodes", 0) or 0))
            model_stats[agent]["avg_evals"].append(float(metrics.get("avg_evals", 0) or 0))
            model_stats[agent]["avg_depth"].append(float(metrics.get("avg_depth", 0) or 0))
            model_stats[agent]["total_turns"].append(float(metrics.get("turns", 0) or 0))
            model_stats[agent]["avg_pip_sum"].append(float((summary_row or {}).get("avg_pip_sum", 0) or 0))

    rows = []
    for agent, stats in model_stats.items():
        rows.append(
            {
                "agent": agent,
                "win_rate": round(sum(stats["win_rate"]) / len(stats["win_rate"]), 4) if stats["win_rate"] else 0,
                "avg_time_ms": round(sum(stats["avg_time_ms"]) / len(stats["avg_time_ms"]), 4) if stats["avg_time_ms"] else 0,
                "avg_nodes": round(sum(stats["avg_nodes"]) / len(stats["avg_nodes"]), 4) if stats["avg_nodes"] else 0,
                "avg_evals": round(sum(stats["avg_evals"]) / len(stats["avg_evals"]), 4) if stats["avg_evals"] else 0,
                "avg_depth": round(sum(stats["avg_depth"]) / len(stats["avg_depth"]), 4) if stats["avg_depth"] else 0,
                "total_turns": round(sum(stats["total_turns"]) / len(stats["total_turns"]), 4) if stats["total_turns"] else 0,
                "avg_pip_sum": round(sum(stats["avg_pip_sum"]) / len(stats["avg_pip_sum"]), 4) if stats["avg_pip_sum"] else 0,
            }
        )
    return rows


def _plot_winrates(csv_path: str, plots_dir: Path) -> dict[str, list[str]]:
    rows = _read_csv_dicts(csv_path)
    if not rows:
        return {}

    plt = _import_matplotlib()
    labels = [r["matchup"] for r in rows]
    win_a = [_maybe_float(r["win_rate_a"]) for r in rows]
    win_b = [_maybe_float(r["win_rate_b"]) for r in rows]
    draws = [_maybe_float(r["draws"]) for r in rows]
    x = list(range(len(labels)))
    width = 0.35

    plt.figure(figsize=(max(8, len(labels) * 1.8), 5))
    plt.bar([i - width / 2 for i in x], win_a, width=width, label="agent_a")
    plt.bar([i + width / 2 for i in x], win_b, width=width, label="agent_b")
    if any(draws):
        plt.plot(x, draws, marker="o", linestyle="--", label="draws")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Win rate / Draws")
    plt.title("Benchmark - Win rates por matchup")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    saved = _save_current_figure(plt, plots_dir / "winrates")
    return {"winrates": saved}


def _plot_model_radar(results: list[dict], plots_dir: Path) -> dict[str, list[str]]:
    rows = _build_model_summary(results)
    if not rows:
        return {}

    plt = _import_matplotlib()

    categories = [
        "win_rate",
        "avg_time_ms",
        "avg_nodes",
        "avg_evals",
        "avg_depth",
        "avg_pip_sum",
    ]
    labels = [
        "WinRate",
        "Time",
        "Nodes",
        "Evals",
        "Depth",
        "Pips",
    ]

    category_values = {cat: [float(r.get(cat, 0) or 0) for r in rows] for cat in categories}

    normalized_by_category = {}
    for cat in categories:
        vals = category_values[cat]
        if cat in {"avg_time_ms", "avg_nodes", "avg_evals", "avg_pip_sum"}:
            max_v = max(vals) if vals else 1
            normalized_by_category[cat] = [1 - (v / max_v if max_v else 0) for v in vals]
        else:
            normalized_by_category[cat] = _normalize_series(vals)

    angles = [n / float(len(categories)) * 2 * math.pi for n in range(len(categories))]
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    for idx, row in enumerate(rows):
        values = [normalized_by_category[cat][idx] for cat in categories]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=row["agent"])
        ax.fill(angles, values, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Radar comparativo multimodal de modelos", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

    saved = _save_current_figure(plt, plots_dir / "radar_multimodal")
    return {"radar_multimodal": saved}


def _plot_avg_metrics_global(csv_path: str, plots_dir: Path) -> dict[str, list[str]]:
    rows = _read_csv_dicts(csv_path)
    if not rows:
        return {}

    plt = _import_matplotlib()
    metric_keys = ["avg_time_ms", "avg_nodes", "avg_evals", "avg_depth", "total_turns"]
    plot_paths = {}

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["agent"]].append(row)

    agents = list(grouped.keys())

    for metric in metric_keys:
        values = []
        for agent in agents:
            vals = [_maybe_float(r.get(metric, 0)) for r in grouped[agent]]
            values.append(sum(vals) / len(vals) if vals else 0)

        x = list(range(len(agents)))
        plt.figure(figsize=(max(8, len(agents) * 1.4), 5))
        plt.bar(x, values)
        plt.xticks(x, agents, rotation=25, ha="right")
        plt.ylabel(metric)
        plt.title(f"Comparativo global - {metric}")
        plt.grid(axis="y", alpha=0.3)
        plot_paths[metric] = _save_current_figure(plt, plots_dir / f"global_{metric}")

    return plot_paths


def _plot_pip_balance_global(csv_path: str, plots_dir: Path) -> dict[str, list[str]]:
    rows = _read_csv_dicts(csv_path)
    if not rows:
        return {}

    plt = _import_matplotlib()
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["agent"]].append(_maybe_float(row["pip_sum"]))

    agents = list(grouped.keys())
    values = [sum(v) / len(v) if v else 0 for v in (grouped[a] for a in agents)]

    x = list(range(len(agents)))
    plt.figure(figsize=(max(8, len(agents) * 1.4), 5))
    plt.bar(x, values)
    plt.xticks(x, agents, rotation=25, ha="right")
    plt.ylabel("avg_pip_sum")
    plt.title("Comparativo global - balance promedio de pips")
    plt.grid(axis="y", alpha=0.3)

    saved = _save_current_figure(plt, plots_dir / "global_pip_balance")
    return {"global_pip_balance": saved}


def _generate_plots_from_results(results: list[dict], csv_paths: dict[str, str], run_id: str) -> dict[str, list[str]]:
    plots_dir = RESULTS_DIR / PLOTS_DIRNAME / run_id
    plots_dir.mkdir(parents=True, exist_ok=True)

    generated: dict[str, list[str]] = {}
    try:
        for key, value in _plot_winrates(csv_paths["winrates"], plots_dir).items():
            generated[key] = value

        for key, value in _plot_model_radar(results, plots_dir).items():
            generated[key] = value

        for key, value in _plot_avg_metrics_global(csv_paths["avg_metrics"], plots_dir).items():
            generated[key] = value

        for key, value in _plot_pip_balance_global(csv_paths["pip_balance"], plots_dir).items():
            generated[key] = value

    except Exception as exc:
        generated["plot_error"] = [str(exc)]

    return generated


def _export_benchmark_csv(results: list[dict], run_id: str) -> dict[str, str]:
    RESULTS_DIR.mkdir(exist_ok=True)
    paths: dict[str, str] = {}

    paths["winrates"] = _export_csv(
        RESULTS_DIR / f"winrates_{run_id}.csv",
        [
            "matchup",
            "agent_a",
            "agent_b",
            "wins_a",
            "wins_b",
            "draws",
            "win_rate_a",
            "win_rate_b",
            "n_games",
            "avg_turns",
        ],
        [
            [
                r["label"],
                r["agent_a"],
                r["agent_b"],
                r["wins_a"],
                r["wins_b"],
                r["draws"],
                r["win_rate_a"],
                r["win_rate_b"],
                r["n_games"],
                r["avg_turns"],
            ]
            for r in results
        ],
    )

    avg_rows = []
    for r in results:
        for role, key in [("A", "metrics_a"), ("B", "metrics_b")]:
            m = r[key] or {}
            agent_name = r["agent_a"] if role == "A" else r["agent_b"]

            avg_rows.append(
                [
                    r["label"],
                    agent_name,
                    role,
                    _round(m.get("avg_time_ms", 0), 4),
                    _round(m.get("avg_nodes", 0), 4),
                    _round(m.get("avg_evals", 0), 4),
                    _round(m.get("avg_depth", 0), 4),
                    int(m.get("turns", 0) or 0),
                ]
            )

    paths["avg_metrics"] = _export_csv(
        RESULTS_DIR / f"avg_metrics_{run_id}.csv",
        [
            "matchup",
            "agent",
            "role",
            "avg_time_ms",
            "avg_nodes",
            "avg_evals",
            "avg_depth",
            "total_turns",
        ],
        avg_rows,
    )

    pip_balance_rows = []
    for r in results:
        for item in r["game_results"]:
            pip_balance_rows.append(
                [r["label"], item["game_index"], r["agent_a"], "A", item["pip_sum_a"]]
            )
            pip_balance_rows.append(
                [r["label"], item["game_index"], r["agent_b"], "B", item["pip_sum_b"]]
            )

    paths["pip_balance"] = _export_csv(
        RESULTS_DIR / f"pip_balance_{run_id}.csv",
        ["matchup", "game_index", "agent", "role", "pip_sum"],
        pip_balance_rows,
    )

    return paths


@router.get("/matchups")
def get_default_matchups():
    return {
        "matchups": [
            {"tag": tag, "agent_a": a, "agent_b": b, "label": label}
            for tag, a, b, label in DEFAULT_TOURNAMENT
        ]
    }


@router.post("/run")
async def run_benchmark(request: BenchmarkRequest):
    """
    Ejecuta el torneo y emite eventos SSE de progreso.
    Cada matchup emite un evento cuando termina.
    Al final emite el resumen completo y exporta los CSVs y gráficas.
    """
    matchups = request.matchups or [
        {"tag": tag, "agent_a": a, "agent_b": b, "label": label}
        for tag, a, b, label in DEFAULT_TOURNAMENT
    ]
    n_games = request.n_games
    run_id = str(uuid.uuid4())[:8]

    async def event_generator():
        loop = asyncio.get_event_loop()
        t0 = time.time()
        all_results = []

        yield f"data: {json.dumps({'type': 'start', 'run_id': run_id, 'n_matchups': len(matchups), 'n_games': n_games})}\n\n"

        for idx, m in enumerate(matchups):
            tag = m.get("tag", f"matchup_{idx}")
            na = m["agent_a"]
            nb = m["agent_b"]
            label = m.get("label", f"{na} vs {nb}")

            yield f"data: {json.dumps({'type': 'matchup_start', 'index': idx, 'tag': tag, 'label': label})}\n\n"

            result = await loop.run_in_executor(
                None, _run_matchup, tag, na, nb, label, n_games
            )
            all_results.append(result)

            yield f"data: {json.dumps({'type': 'matchup_done', 'index': idx, **result})}\n\n"

        total_time = round(time.time() - t0, 2)
        exported_paths = _export_benchmark_csv(all_results, run_id)
        plot_paths = await loop.run_in_executor(
            None, _generate_plots_from_results, all_results, exported_paths, run_id
        )

        yield f"data: {json.dumps({'type': 'benchmark_done', 'run_id': run_id, 'total_time_s': total_time, 'results': all_results, 'exported_files': exported_paths, 'plot_files': plot_paths})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )