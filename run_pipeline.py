"""Run fetch -> build_dataset -> train_and_evaluate in order."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(script: str) -> None:
    subprocess.run([sys.executable, str(ROOT / script)], check=True)


if __name__ == "__main__":
    run("fetch_data.py")
    run("build_dataset.py")
    run("train_and_evaluate.py")
