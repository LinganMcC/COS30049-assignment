"""Safe development runner.

This deliberately stops after TRAIN inspection + coarse TRAIN CV tuning.
It never evaluates DRCAT TEST or HC3.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

for script in ("01_inspect_train.py", "02_coarse_tuning.py"):
    print(f"\n{'=' * 72}\n{script}\n{'=' * 72}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True)

print("\nDevelopment stage complete.")
print("Inspect outputs/02_coarse_tuning/configuration_summary.csv before fine tuning.")
