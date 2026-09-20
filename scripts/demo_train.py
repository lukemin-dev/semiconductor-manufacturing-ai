from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import make_demo_dataset
from src.train import train_all

if __name__ == "__main__":
    print(
        "WARNING: training on generated demo data; metrics are NOT portfolio results."
    )
    result = train_all(make_demo_dataset(), ROOT / "artifacts_demo")
    print(result)
