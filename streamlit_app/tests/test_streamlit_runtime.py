"""Runtime regression tests for the Streamlit dashboard."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
PAGES = [
    "app.py",
    "pages/1_Care_Pipeline_Flow.py",
    "pages/2_Transfer_Discharge_Efficiency.py",
    "pages/3_Bottleneck_Detection.py",
    "pages/4_Outcome_Trend_Analysis.py",
    "pages/5_ML_Forecast_and_Risk.py",
]


class StreamlitRuntimeTests(unittest.TestCase):
    def test_dashboard_pages_run_without_streamlit_compatibility_warnings(self) -> None:
        """Every page should execute cleanly with the installed Streamlit version."""
        script = textwrap.dedent(
            f"""
            from streamlit.testing.v1 import AppTest

            for path in {PAGES!r}:
                app = AppTest.from_file(path)
                timeout = 180 if "ML_Forecast" in path else 30
                app.run(timeout=timeout)
                assert not app.exception, (path, app.exception)
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=APP_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout + result.stderr

        self.assertEqual(result.returncode, 0, output)
        self.assertNotIn("default value but also had its value set via the Session State API", output)
        self.assertNotIn("Please replace `use_container_width` with `width`", output)


if __name__ == "__main__":
    unittest.main()
