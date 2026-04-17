"""
Integration tests for FuturesFBWorkspace.execute().

Each test spawns a real Python subprocess in a temporary directory, so these
tests are slower than the pure-unit tests.  No TX market data is required;
all factor.py scripts either generate synthetic data or test error paths.

Fixtures
--------
patch_settings : overrides FUTURES_COSTEER_SETTINGS so that
  - data_folder_debug → an empty temp dir (no symlinks needed)
  - python_bin        → sys.executable (the current interpreter)
  - file_based_execution_timeout → 30 s (keeps tests fast)
"""

import sys
import textwrap

import pandas as pd
import pytest

from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.scenarios.futures.experiment import FuturesFBWorkspace


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def patch_settings(monkeypatch, tmp_path):
    """Redirect data folders and python binary for isolated subprocess tests."""
    data_dir = tmp_path / "mock_data"
    data_dir.mkdir()

    from rdagent.scenarios.futures.developer.factor_coder import FuturesCoSTEERSettings

    custom = FuturesCoSTEERSettings(
        data_folder=str(data_dir),
        data_folder_debug=str(data_dir),
        python_bin=sys.executable,
        file_based_execution_timeout=30,
    )
    monkeypatch.setattr(
        "rdagent.scenarios.futures.developer.factor_coder.FUTURES_COSTEER_SETTINGS",
        custom,
    )


def _task(name: str = "test_signal") -> FactorTask:
    return FactorTask(factor_name=name, factor_description="test", factor_formulation="test")


def _ws(tmp_path, code: str, name: str = "test_signal") -> FuturesFBWorkspace:
    """Build a workspace with the given factor.py code rooted at tmp_path."""
    ws = FuturesFBWorkspace(target_task=_task(name))
    ws.workspace_path = tmp_path / name
    ws.inject_files(**{"factor.py": textwrap.dedent(code)})
    return ws


# ── Tests: early-exit paths ───────────────────────────────────────────────────


class TestWorkspaceEarlyExit:
    def test_no_code_returns_code_not_set(self):
        ws = FuturesFBWorkspace(target_task=_task())
        feedback, result = ws.execute()
        assert feedback == FuturesFBWorkspace.CODE_NOT_SET
        assert result is None


# ── Tests: successful execution ───────────────────────────────────────────────


class TestWorkspaceValidCode:
    def test_valid_series_returns_series_result(self, tmp_path):
        code = """
            import pandas as pd, numpy as np
            idx = pd.date_range("2023-01-03 08:45", periods=10, freq="1min")
            signal = pd.Series(np.ones(10), index=idx, name="signal")
            signal.to_hdf("result.h5", key="signal", mode="w")
        """
        ws = _ws(tmp_path, code)
        feedback, result = ws.execute()

        assert FuturesFBWorkspace.EXEC_SUCCESS in feedback
        assert isinstance(result, pd.Series)
        assert result.name == "signal"
        assert len(result) == 10

    def test_valid_series_no_nan(self, tmp_path):
        code = """
            import pandas as pd, numpy as np
            idx = pd.date_range("2023-01-03 08:45", periods=5, freq="1min")
            signal = pd.Series([1.0, -1.0, 0.0, 1.0, -1.0], index=idx, name="signal")
            signal.to_hdf("result.h5", key="signal", mode="w")
        """
        ws = _ws(tmp_path, code)
        _, result = ws.execute()
        assert result is not None
        assert result.isna().sum() == 0

    def test_single_column_df_squeezed_to_series(self, tmp_path):
        """
        The type-guard block in the prompt (squeeze + assert) converts a
        single-column DataFrame to a Series before saving.
        """
        code = """
            import pandas as pd, numpy as np
            idx = pd.date_range("2023-01-03 08:45", periods=5, freq="1min")
            # Intermediate single-column DataFrame (common LLM pattern)
            signal = pd.DataFrame({"signal": np.ones(5)}, index=idx)

            # --- type guard copied from prompt template ---
            if isinstance(signal, pd.DataFrame):
                signal = signal.squeeze()
            assert isinstance(signal, pd.Series), (
                f"signal must be pd.Series before saving, got {type(signal)}"
            )
            signal = signal.fillna(0)
            signal.name = "signal"
            signal.to_hdf("result.h5", key="signal", mode="w")
        """
        ws = _ws(tmp_path, code)
        feedback, result = ws.execute()

        assert isinstance(result, pd.Series), f"Expected Series, got {type(result)}"
        assert result.name == "signal"

    def test_output_file_present_marker_in_feedback(self, tmp_path):
        code = """
            import pandas as pd, numpy as np
            idx = pd.date_range("2023-01-03 08:45", periods=3, freq="1min")
            signal = pd.Series(np.zeros(3), index=idx, name="signal")
            signal.to_hdf("result.h5", key="signal", mode="w")
        """
        ws = _ws(tmp_path, code)
        feedback, _ = ws.execute()
        assert FuturesFBWorkspace.OUTPUT_FOUND in feedback


# ── Tests: failure paths ──────────────────────────────────────────────────────


class TestWorkspaceFailurePaths:
    def test_syntax_error_returns_none_result(self, tmp_path):
        code = "this is not valid python @@@ syntax error ###"
        ws = _ws(tmp_path, code)
        feedback, result = ws.execute()

        assert result is None
        assert FuturesFBWorkspace.OUTPUT_NOT_FOUND in feedback

    def test_runtime_exception_returns_none_result(self, tmp_path):
        code = """
            raise RuntimeError("deliberate failure")
        """
        ws = _ws(tmp_path, code)
        feedback, result = ws.execute()

        assert result is None
        assert "RuntimeError" in feedback

    def test_multi_column_df_triggers_assertion_error(self, tmp_path):
        """
        Saving a multi-column DataFrame with the type guard active must raise
        AssertionError because squeeze() returns a DataFrame, not a Series.
        """
        code = """
            import pandas as pd, numpy as np
            idx = pd.date_range("2023-01-03 08:45", periods=5, freq="1min")
            # Multi-column DataFrame — cannot be squeezed to Series
            signal = pd.DataFrame({"a": np.ones(5), "b": np.zeros(5)}, index=idx)

            if isinstance(signal, pd.DataFrame):
                signal = signal.squeeze()
            assert isinstance(signal, pd.Series), (
                f"signal must be pd.Series before saving, got {type(signal)}"
            )
            signal.to_hdf("result.h5", key="signal", mode="w")
        """
        ws = _ws(tmp_path, code)
        feedback, result = ws.execute()

        assert result is None
        assert "AssertionError" in feedback

    def test_missing_result_h5_returns_none(self, tmp_path):
        code = """
            # Code runs successfully but never writes result.h5
            x = 1 + 1
        """
        ws = _ws(tmp_path, code)
        feedback, result = ws.execute()

        assert result is None
        assert FuturesFBWorkspace.OUTPUT_NOT_FOUND in feedback

    def test_nan_values_preserved_without_fillna(self, tmp_path):
        """
        If generated code skips fillna(), the returned Series contains NaN.
        The workspace itself doesn't enforce this — the runner/evaluator does.
        """
        code = """
            import pandas as pd, numpy as np
            idx = pd.date_range("2023-01-03 08:45", periods=5, freq="1min")
            signal = pd.Series([1.0, np.nan, 1.0, np.nan, 1.0], index=idx, name="signal")
            signal.to_hdf("result.h5", key="signal", mode="w")
        """
        ws = _ws(tmp_path, code)
        feedback, result = ws.execute()

        assert isinstance(result, pd.Series)
        assert result.isna().sum() == 2  # NaN values are preserved, not auto-filled


# ── Tests: all_codes property ─────────────────────────────────────────────────


class TestAllCodesProperty:
    def test_all_codes_returns_injected_code(self, tmp_path):
        code = "# my special factor\nimport pandas as pd\n"
        ws = _ws(tmp_path, code)
        assert "my special factor" in ws.all_codes

    def test_all_codes_empty_before_inject(self):
        ws = FuturesFBWorkspace(target_task=_task())
        assert ws.all_codes == ""
