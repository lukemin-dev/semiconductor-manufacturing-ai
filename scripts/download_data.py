from pathlib import Path
import io
import zipfile
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)

URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"


def main():
    print(f"Downloading UCI SECOM dataset from {URL}")
    r = requests.get(URL, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        wanted = {"secom.data", "secom_labels.data", "secom.names"}
        names = {Path(n).name: n for n in zf.namelist()}
        for filename in wanted:
            if filename in names:
                target = OUT / filename
                target.write_bytes(zf.read(names[filename]))
                print("saved", target)
    print("done")


if __name__ == "__main__":
    main()
