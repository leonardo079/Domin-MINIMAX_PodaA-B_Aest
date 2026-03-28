import time
import csv
import os
from dataclasses import dataclass, field


@dataclass
class TurnMetrics:
    strategy: str
    turn: int
    time_ms: float
    nodes_expanded: int
    eval_calls: int
    max_depth: int
    move_chosen: str

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "turn": self.turn,
            "time_ms": self.time_ms,
            "nodes_expanded": self.nodes_expanded,
            "eval_calls": self.eval_calls,
            "max_depth": self.max_depth,
            "move_chosen": self.move_chosen,
        }


class CostProfiler:
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.metrics: list[TurnMetrics] = []
        self._start_time = None
        self.nodes_expanded = 0
        self.eval_calls = 0
        self.max_depth = 0
        self._turn = 0

    def start_turn(self):
        self._start_time = time.perf_counter()
        self.nodes_expanded = 0
        self.eval_calls = 0
        self.max_depth = 0

    def count_node(self):
        self.nodes_expanded += 1

    def count_eval(self):
        self.eval_calls += 1

    def update_depth(self, d: int):
        if d > self.max_depth:
            self.max_depth = d

    def end_turn(self, move_str: str) -> TurnMetrics:
        elapsed = (time.perf_counter() - self._start_time) * 1000
        self._turn += 1
        m = TurnMetrics(
            strategy=self.strategy_name,
            turn=self._turn,
            time_ms=round(elapsed, 3),
            nodes_expanded=self.nodes_expanded,
            eval_calls=self.eval_calls,
            max_depth=self.max_depth,
            move_chosen=move_str,
        )
        self.metrics.append(m)
        return m

    def last_metric_dict(self) -> dict | None:
        if not self.metrics:
            return None
        return self.metrics[-1].to_dict()

    def summary(self) -> dict:
        if not self.metrics:
            return {}
        times = [m.time_ms for m in self.metrics]
        nodes = [m.nodes_expanded for m in self.metrics]
        evals = [m.eval_calls for m in self.metrics]
        depths = [m.max_depth for m in self.metrics]
        return {
            "strategy": self.strategy_name,
            "turns": len(self.metrics),
            "avg_time_ms": round(sum(times) / len(times), 3),
            "max_time_ms": round(max(times), 3),
            "total_time_ms": round(sum(times), 3),
            "avg_nodes": round(sum(nodes) / len(nodes), 1),
            "total_nodes": sum(nodes),
            "avg_evals": round(sum(evals) / len(evals), 1),
            "total_evals": sum(evals),
            "avg_depth": round(sum(depths) / len(depths), 1),
            "max_depth_ever": max(depths) if depths else 0,
        }

    def all_metrics_list(self) -> list[dict]:
        return [m.to_dict() for m in self.metrics]

    def export_csv(self, path: str = "results/metrics.csv"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        file_exists = os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "strategy", "turn", "time_ms", "nodes_expanded",
                "eval_calls", "max_depth", "move_chosen",
            ])
            if not file_exists:
                writer.writeheader()
            for m in self.metrics:
                writer.writerow(m.__dict__)
