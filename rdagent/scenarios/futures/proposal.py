"""
Hypothesis generation and Hypothesis→Experiment conversion for TX futures.

Reuses LLMHypothesisGen / LLMHypothesis2Experiment from the components layer;
only the context-preparation and JSON-parsing methods are futures-specific.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from collections import Counter
from typing import Tuple

from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.components.proposal import FactorHypothesis2Experiment, FactorHypothesisGen
from rdagent.core.proposal import Hypothesis, Scenario, Trace
from rdagent.scenarios.futures.experiment import FuturesFactorExperiment
from rdagent.utils.agent.tpl import T


class FuturesFactorHypothesis(Hypothesis):
    """Hypothesis extended with a method_type label for exhaustion tracking."""

    def __init__(self, *args, method_type: str = "other", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.method_type: str = method_type


# ─────────────────────── HypothesisGen ──────────────────────────────────


def _format_full_trace_for_meta_llm(trace: Trace) -> str:
    """Render the full trace into a compact text for the meta-summarization LLM call."""
    blocks = []
    for i, (exp, fb) in enumerate(trace.hist, 1):
        hyp = getattr(exp, "hypothesis", None)
        mt = getattr(hyp, "method_type", "unknown") if hyp else "unknown"
        result = exp.result or {}
        sharpe = result.get("sharpe", "N/A")
        decision = getattr(fb, "decision", False)
        obs = getattr(fb, "observations", "")

        signals = []
        for task in getattr(exp, "sub_tasks", []):
            name = getattr(task, "factor_name", "")
            formulation = getattr(task, "factor_formulation", "")
            if name:
                signals.append(f"  - {name}: {formulation[:120]}" if formulation else f"  - {name}")

        block = (
            f"### Trial {i} [{mt}] | Sharpe={sharpe} | SOTA={decision}\n"
            + ("\n".join(signals) if signals else "  (no signals)")
            + f"\nFeedback: {str(obs)[:200]}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def _search_arxiv_for_futures(method_types: list[str], max_results: int = 5) -> str:
    """Query arxiv API for recent papers on index futures intraday signals.

    Returns formatted abstracts, or empty string on any network/parse error.
    """
    import xml.etree.ElementTree as ET

    import requests

    keywords = "index+futures+intraday+signal+microstructure"
    url = (
        "http://export.arxiv.org/api/query"
        f"?search_query=ti:{keywords}"
        f"&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = []
        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            if title_el is None or summary_el is None:
                continue
            title = title_el.text.strip().replace("\n", " ")
            summary = summary_el.text.strip().replace("\n", " ")[:300]
            entries.append(f"- **{title}**\n  {summary}")
        if not entries:
            return ""
        return "### Recent arxiv Research (abstracts)\n" + "\n".join(entries)
    except Exception:
        return ""  # never block hypothesis generation on network issues


def _generate_strategy_map(trace: Trace) -> str:
    """Call LLM to produce a structured strategy map from the full trace.

    Also queries arxiv for recent papers to seed the 'encouraged directions' section.
    """
    from rdagent.oai.llm_utils import APIBackend

    active_methods = list({
        getattr(getattr(exp, "hypothesis", None), "method_type", None)
        for exp, _ in trace.hist[-10:]
        if getattr(getattr(exp, "hypothesis", None), "method_type", None)
    })
    web_context = _search_arxiv_for_futures(active_methods)
    full_trace = _format_full_trace_for_meta_llm(trace)
    system_prompt = T("scenarios.futures.prompts:strategy_map_generation.system").r()
    user_prompt = T("scenarios.futures.prompts:strategy_map_generation.user").r(
        full_trace=full_trace,
        web_context=web_context,
    )
    return APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt, json_mode=False)


def _build_method_type_summary(trace: Trace) -> str:
    """
    Group trace.hist by method_type and return a compact summary.

    For each method_type, shows: attempt count, best Sharpe, SOTA wins,
    and the most recent feedback observation — so the LLM understands the
    landscape without reading every flat trial in sequence.
    """
    from collections import defaultdict

    groups: dict[str, list] = defaultdict(list)
    for exp, fb in trace.hist:
        mt = getattr(getattr(exp, "hypothesis", None), "method_type", None) or "other"
        groups[mt].append((exp, fb))

    lines = ["## Method-Type Summary (all rounds grouped)"]
    for mt, trials in groups.items():
        sharpes = [
            t[0].result.get("sharpe")
            for t in trials
            if t[0].result and t[0].result.get("sharpe") is not None
        ]
        best_sharpe = f"{max(sharpes):.3f}" if sharpes else "N/A"
        wins = sum(1 for _, fb in trials if fb and fb.decision)
        last_obs = ""
        for _, fb in reversed(trials):
            obs = getattr(fb, "observations", None)
            if obs:
                last_obs = obs[:200]
                break
        lines.append(
            f"\n### {mt}  (tried {len(trials)}x | best Sharpe {best_sharpe} | SOTA wins {wins})\n"
            f"Last observation: {last_obs}"
        )
    return "\n".join(lines)


def _compute_exhausted_methods(trace: Trace, max_tries: int) -> list[str]:
    """
    Return method_types that have been tried >= max_tries times with zero SOTA wins.

    A method_type is exhausted when it appears at least max_tries times in the
    trace history AND none of those trials had feedback.decision == True.
    """
    tries: Counter = Counter()
    wins: Counter = Counter()
    for exp, fb in trace.hist:
        mt = getattr(getattr(exp, "hypothesis", None), "method_type", None)
        if mt:
            tries[mt] += 1
            if fb and fb.decision:
                wins[mt] += 1
    return [mt for mt, count in tries.items() if count >= max_tries and wins[mt] == 0]


STRATEGY_MAP_UPDATE_INTERVAL = 5  # re-generate the strategy map every N rounds


class FuturesFactorHypothesisGen(FactorHypothesisGen):
    """
    Proposes a new TX 1-min signal idea based on the experiment history (trace).
    """

    def __init__(self, scen: Scenario) -> None:
        super().__init__(scen)
        self._strategy_map: str = ""        # cached LLM-generated strategy map
        self._strategy_map_round: int = -1  # trace.hist length when map was last generated
        self._paper_kb = None               # FuturesPaperKnowledgeBase, loaded lazily
        self._load_paper_kb()

    def _load_paper_kb(self) -> None:
        from pathlib import Path

        from rdagent.app.futures_rd_loop.conf import FUTURES_FACTOR_PROP_SETTING
        from rdagent.scenarios.futures.knowledge_management.paper_kb import (
            FuturesPaperKnowledgeBase,
        )

        cfg = FUTURES_FACTOR_PROP_SETTING
        if not cfg.paper_folder:
            return

        cache = Path(cfg.paper_kb_cache) if cfg.paper_kb_cache else None
        kb = FuturesPaperKnowledgeBase(path=cache)
        # PDVectorBase.load() is called in __init__ when path exists; if cache
        # already had content, vector_df will be populated after load().
        if kb.shape()[0] == 0:
            kb.build_from_folder(cfg.paper_folder)
            if cache:
                kb.dump()
        self._paper_kb = kb

    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        from rdagent.app.futures_rd_loop.conf import FUTURES_FACTOR_PROP_SETTING

        if not trace.hist:
            hypothesis_and_feedback = "No previous experiments yet – this is the first round."
        else:
            n_hist = len(trace.hist)
            # Regenerate the strategy map every STRATEGY_MAP_UPDATE_INTERVAL rounds,
            # or on the very first round that has enough history to summarize.
            rounds_since_update = n_hist - self._strategy_map_round
            if not self._strategy_map or rounds_since_update >= STRATEGY_MAP_UPDATE_INTERVAL:
                self._strategy_map = _generate_strategy_map(trace)
                self._strategy_map_round = n_hist

            # Show the LLM-generated strategy map + recent N trials in detail.
            recent_n = 5
            recent_hist = SimpleNamespace(hist=trace.hist[-recent_n:])
            recent_detail = T("scenarios.futures.prompts:hypothesis_and_feedback").r(trace=recent_hist)
            hypothesis_and_feedback = (
                f"{self._strategy_map}\n\n"
                f"## Recent {min(recent_n, n_hist)} Trials (detailed)\n"
                f"{recent_detail}"
            )

        last_hypothesis_and_feedback = (
            T("scenarios.futures.prompts:last_hypothesis_and_feedback").r(
                experiment=trace.hist[-1][0],
                feedback=trace.hist[-1][1],
            )
            if trace.hist
            else "No previous experiments yet."
        )

        n_rounds = len(trace.hist)
        rag_hint = (
            "Start with simple, interpretable signals (momentum, VWAP deviation, volume breakout)."
            if n_rounds < 10
            else "The simple signals have been explored. Try more complex combinations or ML-based signals."
        )

        exhausted = _compute_exhausted_methods(trace, FUTURES_FACTOR_PROP_SETTING.max_method_tries)
        if exhausted:
            exhausted_note = (
                f"\n\nEXHAUSTED METHOD TYPES (do NOT propose these again — "
                f"tried {FUTURES_FACTOR_PROP_SETTING.max_method_tries}+ times without SOTA improvement): "
                + ", ".join(exhausted)
            )
            rag_hint += exhausted_note

        # Prepend relevant paper excerpts to rag_hint when paper KB is loaded.
        if self._paper_kb is not None:
            paper_ctx = self._paper_kb.retrieve(
                query=f"Taiwan index futures 1-minute intraday signal {rag_hint[:100]}",
                topk=3,
            )
            if paper_ctx:
                rag_hint = paper_ctx + "\n\n" + rag_hint

        return {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "last_hypothesis_and_feedback": last_hypothesis_and_feedback,
            "RAG": rag_hint,
            "hypothesis_output_format": T("scenarios.futures.prompts:hypothesis_output_format").r(),
            "hypothesis_specification": T("scenarios.futures.prompts:hypothesis_specification").r(),
        }, True

    def convert_response(self, response: str) -> Hypothesis:
        d = json.loads(response)
        return FuturesFactorHypothesis(
            hypothesis=d.get("hypothesis", ""),
            reason=d.get("reason", ""),
            concise_reason=d.get("concise_reason", ""),
            concise_observation=d.get("concise_observation", ""),
            concise_justification=d.get("concise_justification", ""),
            concise_knowledge=d.get("concise_knowledge", ""),
            method_type=d.get("method_type", "other"),
        )


class FuturesFactorHypothesisGenFromStrategy(FuturesFactorHypothesisGen):
    """Seeds the first round with a user-provided strategy text, then iterates normally."""

    def __init__(self, scen: Scenario) -> None:
        super().__init__(scen)
        from rdagent.app.futures_rd_loop.conf import FUTURES_FACTOR_PROP_SETTING
        self.initial_strategy: str = FUTURES_FACTOR_PROP_SETTING.initial_strategy

    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        ctx, system = super().prepare_context(trace)
        if not self.initial_strategy:
            return ctx, system

        if not trace.hist:
            seed_text = (
                f"User-provided strategy direction:\n{self.initial_strategy}\n\n"
                "Use this as inspiration to propose a concrete, testable signal hypothesis. "
                "You may interpret it broadly — the goal is to explore the spirit of this "
                "direction, not to implement it literally."
            )
            ctx["hypothesis_and_feedback"] = seed_text
            ctx["last_hypothesis_and_feedback"] = seed_text
            ctx["RAG"] = (
                f"The user has suggested a strategy direction (below). "
                f"Let it guide your thinking, but feel free to adapt or generalise it.\n\n"
                f"--- User strategy ---\n{self.initial_strategy}\n--- End ---"
            )
        else:
            ctx["RAG"] = (
                f"The user's initial strategy direction (below) is a soft preference — "
                f"lean towards it when proposing new hypotheses, but do not feel constrained "
                f"if the experiment history points to a more promising path.\n\n"
                f"--- User strategy ---\n{self.initial_strategy}\n--- End ---\n\n"
                + ctx.get("RAG", "")
            )
        return ctx, system


# ──────────────────── Hypothesis2Experiment ──────────────────────────────

class FuturesFactorHypothesis2Experiment(FactorHypothesis2Experiment):
    """
    Translates a Hypothesis into a FuturesFactorExperiment with FactorTask list.
    """

    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict, bool]:
        if trace.hist:
            hypothesis_and_feedback = T(
                "scenarios.futures.prompts:hypothesis_and_feedback"
            ).r(trace=trace)
        else:
            hypothesis_and_feedback = "No previous experiments yet."

        return {
            "target_hypothesis": str(hypothesis),
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "experiment_output_format": T(
                "scenarios.futures.prompts:factor_experiment_output_format"
            ).r(),
            "target_list": [],
            "RAG": None,
        }, True

    def convert_response(
        self, response: str, hypothesis: Hypothesis, trace: Trace
    ) -> FuturesFactorExperiment:
        from rdagent.app.futures_rd_loop.conf import FUTURES_FACTOR_PROP_SETTING
        from rdagent.scenarios.futures.experiment import FuturesFBWorkspace

        d = json.loads(response)
        tasks = []
        for signal_name, info in d.items():
            tasks.append(
                FactorTask(
                    factor_name=signal_name,
                    factor_description=info.get("description", ""),
                    factor_formulation=info.get("formulation", ""),
                    variables=info.get("variables", {}),
                )
            )

        exp = FuturesFactorExperiment(tasks, hypothesis=hypothesis)

        # Build the baseline: use custom factor.py if provided, else empty zero-signal baseline
        baseline_code = FUTURES_FACTOR_PROP_SETTING.baseline_factor
        if baseline_code:
            baseline_task = FactorTask(
                factor_name="custom_baseline",
                factor_description="User-provided baseline strategy",
                factor_formulation="See factor.py code",
                variables={},
            )
            baseline_ws = FuturesFBWorkspace(target_task=baseline_task)
            baseline_ws.inject_files(**{"factor.py": baseline_code})
            baseline_exp = FuturesFactorExperiment(
                sub_tasks=[baseline_task], hypothesis=None
            )
            baseline_exp.sub_workspace_list = [baseline_ws]
        else:
            baseline_exp = FuturesFactorExperiment(sub_tasks=[])

        # Chain of successful experiments (SOTA history) for the runner to reference
        exp.based_experiments = [baseline_exp] + [
            t[0]
            for t in trace.hist
            if t[1] and isinstance(t[0], FuturesFactorExperiment)
        ]

        # De-duplicate: skip tasks already explored in previous experiments
        seen_names = {
            sub_task.factor_name
            for based_exp in exp.based_experiments
            for sub_task in based_exp.sub_tasks
        }
        exp.sub_tasks = [t for t in tasks if t.factor_name not in seen_names]

        return exp
