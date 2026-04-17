"""
Tests for FuturesFactorScenario — verifies that the scenario descriptor and
the prompt YAML files contain the expected content.

Notably, these tests validate the type-guard fix added to futures/prompts.yaml:
  - futures_factor_interface  must include the isinstance assert + squeeze hint
  - futures_factor_output_format must warn about DataFrame and mention Series
"""

import pytest

from rdagent.scenarios.futures.experiment import FuturesFactorScenario


@pytest.fixture(scope="module")
def scen() -> FuturesFactorScenario:
    return FuturesFactorScenario()


# ── Basic non-empty checks ────────────────────────────────────────────────────


class TestScenarioNonEmpty:
    def test_background_nonempty(self, scen):
        assert len(scen.background.strip()) > 0

    def test_source_data_desc_nonempty(self, scen):
        assert len(scen.get_source_data_desc().strip()) > 0

    def test_interface_nonempty(self, scen):
        assert len(scen.interface.strip()) > 0

    def test_output_format_nonempty(self, scen):
        assert len(scen.output_format.strip()) > 0

    def test_simulator_nonempty(self, scen):
        assert len(scen.simulator.strip()) > 0

    def test_rich_style_description_nonempty(self, scen):
        assert len(scen.rich_style_description.strip()) > 0

    def test_get_runtime_environment_nonempty(self, scen):
        assert len(scen.get_runtime_environment().strip()) > 0


# ── interface prompt: type-guard validation ───────────────────────────────────


class TestInterfaceTypeGuard:
    """The interface must embed the assert/squeeze block so that LLM-generated
    code fails fast when a DataFrame is saved instead of a Series."""

    def test_contains_isinstance_check(self, scen):
        assert "isinstance(signal, pd.Series)" in scen.interface

    def test_contains_assert_statement(self, scen):
        assert "assert isinstance" in scen.interface

    def test_contains_squeeze_hint(self, scen):
        """squeeze() handles the single-column DataFrame → Series edge case."""
        assert "squeeze()" in scen.interface

    def test_contains_fillna_zero(self, scen):
        assert "fillna(0)" in scen.interface

    def test_contains_to_hdf_save_call(self, scen):
        assert "to_hdf" in scen.interface

    def test_contains_signal_name_assignment(self, scen):
        assert 'signal.name = "signal"' in scen.interface

    def test_contains_result_h5_filename(self, scen):
        assert "result.h5" in scen.interface

    def test_contains_data_parquet_loading(self, scen):
        assert "data.parquet" in scen.interface


# ── output_format prompt: Series vs DataFrame warnings ───────────────────────


class TestOutputFormatPrompt:
    def test_mentions_pd_series(self, scen):
        assert "pd.Series" in scen.output_format

    def test_warns_against_dataframe(self, scen):
        """Prompt must explicitly call out that saving a DataFrame is wrong."""
        assert "DataFrame" in scen.output_format

    def test_verification_uses_isinstance(self, scen):
        assert "isinstance(signal, pd.Series)" in scen.output_format

    def test_verification_checks_name(self, scen):
        assert 'signal.name == "signal"' in scen.output_format

    def test_verification_checks_no_nan(self, scen):
        assert "isna().sum() == 0" in scen.output_format

    def test_key_is_signal(self, scen):
        assert 'key="signal"' in scen.output_format


# ── get_scenario_all_desc ─────────────────────────────────────────────────────


class TestGetScenarioAllDesc:
    def test_full_desc_contains_background(self, scen):
        desc = scen.get_scenario_all_desc()
        assert "Background" in desc

    def test_full_desc_contains_source_data(self, scen):
        desc = scen.get_scenario_all_desc()
        assert "data.parquet" in desc

    def test_full_desc_contains_interface(self, scen):
        desc = scen.get_scenario_all_desc()
        assert "isinstance" in desc

    def test_full_desc_contains_output_format(self, scen):
        desc = scen.get_scenario_all_desc()
        assert "pd.Series" in desc

    def test_full_desc_contains_simulator(self, scen):
        desc = scen.get_scenario_all_desc()
        assert "sharpe" in desc.lower()

    def test_simple_background_excludes_interface(self, scen):
        """simple_background=True should return only the background section."""
        desc = scen.get_scenario_all_desc(simple_background=True)
        assert "isinstance" not in desc

    def test_simple_background_still_has_background_text(self, scen):
        desc = scen.get_scenario_all_desc(simple_background=True)
        assert "Background" in desc
