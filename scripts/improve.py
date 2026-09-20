"""Run the predeclared v2 experiment without tuning against the old test."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_secom
from src.experiment import run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-alert-rate", type=float, default=0.30)
    args = parser.parse_args()
    run_experiment(
        load_secom(),
        ROOT / "artifacts",
        ROOT / "experiments/v1/splits.json",
        args.repeats,
        args.max_alert_rate,
    )
