from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"

SECOM_DATA = DATA_RAW / "secom.data"
SECOM_LABELS = DATA_RAW / "secom_labels.data"

RANDOM_STATE = 42
TEST_SIZE = 0.25
MAX_MISSING_RATIO = 0.40
