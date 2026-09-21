"""Run development-only v3 research without changing the deployed v2 model."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_secom
from src.research_v3 import run_v3

if __name__ == "__main__":
    run_v3(load_secom(), ROOT / "artifacts", ROOT / "experiments/v3")
