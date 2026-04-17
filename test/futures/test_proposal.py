"""
Unit tests for the futures proposal layer (no LLM calls).

Covers:
  - FuturesFactorHypothesisGen.prepare_context  — context dict structure,
    empty-trace sentinel, RAG hint evolution, JSON parsing
  - FuturesFactorHypothesis2Experiment.prepare_context — context keys
  - FuturesFactorHypothesis2Experiment.convert_response — experiment creation,
    factor de-duplication across based_experiments
"""

import json
from unittest.mock import Mock, patch

import pytest

from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.core.proposal import Hypothesis, Trace
from rdagent.scenarios.futures.experiment import FuturesFactorExperiment
from rdagent.scenarios.futures.proposal import (
    FuturesFactorHypothesis2Experiment,
    FuturesFactorHypothesisGen,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_scen():
    scen = Mock()
    scen.background = "TX futures background"
    scen.get_scenario_all_desc.return_value = "full scenario description"
    return scen


@pytest.fixture
def empty_trace(mock_scen):
    return Trace(scen=mock_scen)


@pytest.fixture
def base_hypothesis():
    return Hypothesis(
        hypothesis="VWAP deviation predicts intraday reversion",
        reason="price above VWAP tends to mean-revert in TX",
        concise_reason="mean-reversion",
        concise_observation="VWAP deviation pattern",
        concise_justification="empirically valid in equity index futures",
        concise_knowledge="TX intraday range is typically mean-reverting",
    )


def _make_trace_with_history(mock_scen, n: int = 1) -> Trace:
    trace = Trace(scen=mock_scen)
    exp = Mock()
    exp.sub_tasks = [Mock(factor_name="vwap_dev")]
    exp.hypothesis = Mock(action="factor")
    exp.result = {
        "sharpe": 0.8,
        "annual_return_pts": 500.0,
        "max_drawdown_pts": -200.0,
        "n_trades": 100,
        "day_sharpe": 0.6,
        "night_sharpe": 0.4,
    }
    feedback = Mock()
    feedback.observations = "moderate alpha"
    feedback.hypothesis_evaluation = "partially validated"
    feedback.decision = False
    feedback.new_hypothesis = "try momentum"
    feedback.reason = "reversion worked only in day session"
    trace.hist = [(exp, feedback)] * n
    return trace


# ── HypothesisGen ─────────────────────────────────────────────────────────────


class TestFuturesFactorHypothesisGen:
    def test_returns_true_ok_flag(self, mock_scen, empty_trace):
        gen = FuturesFactorHypothesisGen(scen=mock_scen)
        with patch("rdagent.utils.agent.tpl.T.r", return_value="mocked"):
            _, ok = gen.prepare_context(empty_trace)
        assert ok is True

    def test_context_has_rag_key(self, mock_scen, empty_trace):
        gen = FuturesFactorHypothesisGen(scen=mock_scen)
        with patch("rdagent.utils.agent.tpl.T.r", return_value="mocked"):
            ctx, _ = gen.prepare_context(empty_trace)
        assert "RAG" in ctx

    def test_context_has_hypothesis_output_format(self, mock_scen, empty_trace):
        gen = FuturesFactorHypothesisGen(scen=mock_scen)
        with patch("rdagent.utils.agent.tpl.T.r", return_value="mocked"):
            ctx, _ = gen.prepare_context(empty_trace)
        assert "hypothesis_output_format" in ctx

    def test_context_has_hypothesis_specification(self, mock_scen, empty_trace):
        gen = FuturesFactorHypothesisGen(scen=mock_scen)
        with patch("rdagent.utils.agent.tpl.T.r", return_value="mocked"):
            ctx, _ = gen.prepare_context(empty_trace)
        assert "hypothesis_specification" in ctx

    def test_context_has_last_hypothesis_and_feedback(self, mock_scen, empty_trace):
        gen = FuturesFactorHypothesisGen(scen=mock_scen)
        with patch("rdagent.utils.agent.tpl.T.r", return_value="mocked"):
            ctx, _ = gen.prepare_context(empty_trace)
        assert "last_hypothesis_and_feedback" in ctx

    def test_empty_trace_sets_no_experiments_sentinel(self, mock_scen, empty_trace):
        """When there is no history, the context should say so without calling T.r."""
        gen = FuturesFactorHypothesisGen(scen=mock_scen)
        with patch("rdagent.utils.agent.tpl.T.r", return_value="mocked"):
            ctx, _ = gen.prepare_context(empty_trace)
        assert "No previous experiments" in ctx["hypothesis_and_feedback"]

    def test_history_renders_template_for_hypothesis_and_feedback(self, mock_scen):
        trace = _make_trace_with_history(mock_scen, n=1)
        gen = FuturesFactorHypothesisGen(scen=mock_scen)
        with patch("rdagent.utils.agent.tpl.T.r", return_value="rendered_history"):
            ctx, _ = gen.prepare_context(trace)
        # With history, T.r is called → result should be the mocked value
        assert ctx["hypothesis_and_feedback"] == "rendered_history"

    def test_rag_hint_is_simple_before_10_rounds(self, mock_scen):
        trace = _make_trace_with_history(mock_scen, n=5)
        gen = FuturesFactorHypothesisGen(scen=mock_scen)
        with patch("rdagent.utils.agent.tpl.T.r", return_value="mocked"):
            ctx, _ = gen.prepare_context(trace)
        rag = ctx["RAG"]
        # Early rounds hint: simple signals
        assert "simple" in rag.lower() or "momentum" in rag.lower()

    def test_rag_hint_evolves_after_10_rounds(self, mock_scen):
        trace = _make_trace_with_history(mock_scen, n=10)
        gen = FuturesFactorHypothesisGen(scen=mock_scen)
        with patch("rdagent.utils.agent.tpl.T.r", return_value="mocked"):
            ctx, _ = gen.prepare_context(trace)
        rag = ctx["RAG"]
        # Late rounds hint: complex / ML approaches
        assert "complex" in rag.lower() or "ml" in rag.lower()

    def test_convert_response_parses_all_fields(self, mock_scen):
        gen = FuturesFactorHypothesisGen(scen=mock_scen)
        payload = {
            "hypothesis": "short-term momentum persists",
            "reason": "microstructure feedback loops",
            "concise_reason": "momentum",
            "concise_observation": "5-bar return auto-correlation",
            "concise_justification": "observed in TX tick data",
            "concise_knowledge": "TX has thin order book",
        }
        hyp = gen.convert_response(json.dumps(payload))
        assert hyp.hypothesis == payload["hypothesis"]
        assert hyp.reason == payload["reason"]
        assert hyp.concise_reason == payload["concise_reason"]

    def test_convert_response_missing_fields_use_defaults(self, mock_scen):
        gen = FuturesFactorHypothesisGen(scen=mock_scen)
        hyp = gen.convert_response('{"hypothesis": "test"}')
        assert hyp.hypothesis == "test"
        assert hyp.reason == ""  # missing keys default to ""


# ── Hypothesis2Experiment ─────────────────────────────────────────────────────


class TestFuturesFactorHypothesis2Experiment:
    def test_prepare_context_ok_flag(self, empty_trace, base_hypothesis):
        h2e = FuturesFactorHypothesis2Experiment()
        with patch("rdagent.utils.agent.tpl.T.r", return_value="mocked"):
            _, ok = h2e.prepare_context(base_hypothesis, empty_trace)
        assert ok is True

    def test_prepare_context_has_target_hypothesis(self, empty_trace, base_hypothesis):
        h2e = FuturesFactorHypothesis2Experiment()
        with patch("rdagent.utils.agent.tpl.T.r", return_value="mocked"):
            ctx, _ = h2e.prepare_context(base_hypothesis, empty_trace)
        assert "target_hypothesis" in ctx
        assert str(base_hypothesis) in ctx["target_hypothesis"]

    def test_prepare_context_has_experiment_output_format(self, empty_trace, base_hypothesis):
        h2e = FuturesFactorHypothesis2Experiment()
        with patch("rdagent.utils.agent.tpl.T.r", return_value="mocked"):
            ctx, _ = h2e.prepare_context(base_hypothesis, empty_trace)
        assert "experiment_output_format" in ctx

    def test_prepare_context_empty_trace_sentinel(self, empty_trace, base_hypothesis):
        h2e = FuturesFactorHypothesis2Experiment()
        with patch("rdagent.utils.agent.tpl.T.r", return_value="mocked"):
            ctx, _ = h2e.prepare_context(base_hypothesis, empty_trace)
        assert "No previous experiments" in ctx["hypothesis_and_feedback"]

    def test_convert_response_creates_experiment(self, empty_trace, base_hypothesis):
        h2e = FuturesFactorHypothesis2Experiment()
        response = json.dumps({
            "vwap_20": {
                "description": "VWAP deviation over 20 bars",
                "formulation": "(close - vwap) / vwap",
                "variables": {"lookback": "20 bars"},
            }
        })
        exp = h2e.convert_response(response, base_hypothesis, empty_trace)
        assert len(exp.sub_tasks) == 1
        assert exp.sub_tasks[0].factor_name == "vwap_20"

    def test_convert_response_attaches_hypothesis(self, empty_trace, base_hypothesis):
        h2e = FuturesFactorHypothesis2Experiment()
        response = json.dumps({"sig_a": {"description": "d", "formulation": "f", "variables": {}}})
        exp = h2e.convert_response(response, base_hypothesis, empty_trace)
        assert exp.hypothesis is base_hypothesis

    def test_convert_response_empty_json_gives_empty_tasks(self, empty_trace, base_hypothesis):
        h2e = FuturesFactorHypothesis2Experiment()
        exp = h2e.convert_response("{}", base_hypothesis, empty_trace)
        assert exp.sub_tasks == []

    def test_convert_response_multiple_signals(self, empty_trace, base_hypothesis):
        h2e = FuturesFactorHypothesis2Experiment()
        response = json.dumps({
            "sig_a": {"description": "A", "formulation": "a", "variables": {}},
            "sig_b": {"description": "B", "formulation": "b", "variables": {}},
            "sig_c": {"description": "C", "formulation": "c", "variables": {}},
        })
        exp = h2e.convert_response(response, base_hypothesis, empty_trace)
        names = [t.factor_name for t in exp.sub_tasks]
        assert set(names) == {"sig_a", "sig_b", "sig_c"}

    def test_convert_response_deduplicates_seen_factors(self, mock_scen, base_hypothesis):
        """
        Factors already explored in based_experiments must be filtered out
        so the loop does not repeat work.
        """
        trace = Trace(scen=mock_scen)
        prior_exp = FuturesFactorExperiment(
            sub_tasks=[FactorTask(factor_name="vwap_20", factor_description="", factor_formulation="")],
            hypothesis=base_hypothesis,
        )
        trace.hist = [(prior_exp, Mock(decision=False))]

        h2e = FuturesFactorHypothesis2Experiment()
        # vwap_20 already explored; momentum_5 is new
        response = json.dumps({
            "vwap_20": {"description": "dup", "formulation": "dup", "variables": {}},
            "momentum_5": {"description": "new", "formulation": "new", "variables": {}},
        })
        exp = h2e.convert_response(response, base_hypothesis, trace)
        names = [t.factor_name for t in exp.sub_tasks]

        assert "vwap_20" not in names, "Duplicate factor should be filtered"
        assert "momentum_5" in names, "New factor should be included"

    def test_convert_response_all_seen_factors_gives_empty(self, mock_scen, base_hypothesis):
        """If every proposed factor has already been explored, result is empty."""
        trace = Trace(scen=mock_scen)
        prior_exp = FuturesFactorExperiment(
            sub_tasks=[
                FactorTask(factor_name="sig_x", factor_description="", factor_formulation=""),
                FactorTask(factor_name="sig_y", factor_description="", factor_formulation=""),
            ],
            hypothesis=base_hypothesis,
        )
        trace.hist = [(prior_exp, Mock(decision=False))]

        h2e = FuturesFactorHypothesis2Experiment()
        response = json.dumps({
            "sig_x": {"description": "d", "formulation": "f", "variables": {}},
            "sig_y": {"description": "d", "formulation": "f", "variables": {}},
        })
        exp = h2e.convert_response(response, base_hypothesis, trace)
        assert exp.sub_tasks == []
