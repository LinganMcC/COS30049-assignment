"""Explicit final evaluation entry point.

Requires an already frozen model. TEST/HC3 are never touched by run_development.py.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

for script in ("05_final_eval.py", "06_error_analysis.py"):
    print(f"\n{'=' * 72}\n{script}\n{'=' * 72}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True)

print("\nFinal evaluation complete.")
print("Optional stress experiments are separate: python experiments/stress_tests.py")
