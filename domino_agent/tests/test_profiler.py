"""
tests/test_profiler.py — Tests unitarios del CostProfiler.
"""
import pytest
from app.core.profiler import CostProfiler, TurnMetrics


class TestCostProfiler:

    def test_initial_state(self):
        p = CostProfiler("test")
        assert p.metrics == []
        assert p.nodes_expanded == 0

    def test_start_turn_resets_counters(self):
        p = CostProfiler("test")
        p.nodes_expanded = 99
        p.eval_calls = 99
        p.max_depth = 99
        p.start_turn()
        assert p.nodes_expanded == 0
        assert p.eval_calls == 0
        assert p.max_depth == 0

    def test_count_node(self):
        p = CostProfiler("test")
        p.start_turn()
        p.count_node()
        p.count_node()
        assert p.nodes_expanded == 2

    def test_count_eval(self):
        p = CostProfiler("test")
        p.start_turn()
        p.count_eval()
        assert p.eval_calls == 1

    def test_update_depth(self):
        p = CostProfiler("test")
        p.start_turn()
        p.update_depth(3)
        p.update_depth(1)
        p.update_depth(5)
        assert p.max_depth == 5

    def test_end_turn_appends_metric(self):
        p = CostProfiler("test")
        p.start_turn()
        p.count_node()
        p.count_eval()
        p.update_depth(2)
        m = p.end_turn("tile")
        assert len(p.metrics) == 1
        assert isinstance(m, TurnMetrics)
        assert m.strategy == "test"
        assert m.nodes_expanded == 1
        assert m.eval_calls == 1
        assert m.max_depth == 2

    def test_summary_after_turns(self):
        p = CostProfiler("test")
        for _ in range(3):
            p.start_turn()
            p.count_node()
            p.count_node()
            p.count_eval()
            p.update_depth(4)
            p.end_turn("move")

        s = p.summary()
        assert s["turns"] == 3
        assert s["avg_nodes"] == 2.0
        assert s["avg_evals"] == 1.0
        assert s["avg_depth"] == 4.0

    def test_summary_empty(self):
        p = CostProfiler("test")
        assert p.summary() == {}

    def test_all_metrics_list(self):
        p = CostProfiler("test")
        p.start_turn()
        p.end_turn("move")
        result = p.all_metrics_list()
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert "time_ms" in result[0]

    def test_last_metric_dict_none_when_empty(self):
        p = CostProfiler("test")
        assert p.last_metric_dict() is None

    def test_last_metric_dict_after_turn(self):
        p = CostProfiler("test")
        p.start_turn()
        p.end_turn("tile")
        d = p.last_metric_dict()
        assert d is not None
        assert d["strategy"] == "test"
