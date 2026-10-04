"""Smoke-test the cleaned workflow on synthetic data.

Run from logistic_regression/:
    python tests/test_smoke.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from make_synthetic_data import write_all

ROOT = Path(__file__).resolve().parents[1]


def run(script, env, *args):
    subprocess.run([sys.executable, str(script), *args], check=True, env=env)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        data = tmp / "features"
        out = tmp / "outputs"
        write_all(data)

        # Use the real project's sibling feature_extraction folder by default.
        feature_dir = Path(os.environ.get("LR_FEATURE_DIR", ROOT.parent / "feature_extraction"))
        if not (feature_dir / "feature_config.py").exists():
            raise SystemExit(
                "Smoke test needs the project's feature_extraction/feature_config.py. "
                "Set LR_FEATURE_DIR if necessary."
            )

        env = {
            **os.environ,
            "LR_DATA_DIR": str(data),
            "LR_OUTPUT_DIR": str(out),
            "LR_FEATURE_DIR": str(feature_dir),
            "LR_N_BOOTSTRAP": "50",
        }

        run(ROOT / "scripts" / "01_inspect_train.py", env)
        run(ROOT / "scripts" / "02_coarse_tuning.py", env)
        run(ROOT / "scripts" / "03_fine_tuning.py", env)
        run(ROOT / "scripts" / "04_validate_freeze.py", env, "--use-cv-recommendation")
        run(ROOT / "scripts" / "05_final_eval.py", env)
        run(ROOT / "scripts" / "06_error_analysis.py", env, "--n-docs", "1")
        run(ROOT / "experiments" / "stress_tests.py", env, "--min-ai-docs", "2")

        expected = [
            "01_inspection/train_summary.json",
            "02_coarse_tuning/configuration_summary.csv",
            "03_fine_tuning/configuration_summary.csv",
            "04_validation/threshold_sweep.csv",
            "05_frozen_model/logistic_regression.joblib",
            "05_frozen_model/frozen_config.json",
            "06_final_eval/test/metrics_with_group_bootstrap_ci.csv",
            "06_final_eval/hc3/metrics_with_group_bootstrap_ci.csv",
            "06_final_eval/comparison/drcat_vs_hc3.csv",
            "07_error_analysis/test_error_analysis.md",
            "08_experiments/stress_tests.md",
        ]
        missing = [p for p in expected if not (out / p).exists()]
        assert not missing, f"Missing outputs: {missing}"

        frozen = json.loads((out / "05_frozen_model" / "frozen_config.json").read_text())
        assert 0.05 <= frozen["threshold"] <= 0.95

    print("SMOKE TEST PASSED")
