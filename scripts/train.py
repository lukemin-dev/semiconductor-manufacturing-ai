from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_secom
from src.train import train_all


def main():
    dataset = load_secom()
    result = train_all(dataset)
    print("Selected model:", result["best_model"])
    for name, metric in result["metrics"].items():
        print(name, metric)


if __name__ == "__main__":
    main()
