"""Shift the dummy forecast inputs so the last value ends 7 days before today.

Refreshes dummy_predict.json (v1) and dummy_predict_v2.json (v2).

Run with:
    poetry run python scripts/refresh_dummy_predict.py
"""

import datetime as dt
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
TARGET_LAG_DAYS = 7
# Dummy forecast inputs and the timestamp key of their values
DUMMY_PREDICT_FILES = {"dummy_predict.json": "time", "dummy_predict_v2.json": "timestamp"}


def refresh(path: Path, time_key: str) -> None:
    """Shift timestamps in a dummy forecast input to end 7 days before today."""
    with open(path) as f:
        d = json.load(f)

    vals = d["target_series"]["historical_values"]
    last_time = dt.datetime.fromisoformat(vals[-1][time_key])
    target_end = dt.datetime.now() - dt.timedelta(days=TARGET_LAG_DAYS)
    shift = dt.timedelta(days=(target_end.date() - last_time.date()).days)

    for v in vals:
        v[time_key] = (dt.datetime.fromisoformat(v[time_key]) + shift).isoformat()

    with open(path, "w") as f:
        json.dump(d, f, indent=2)
        f.write("\n")

    print(f"Refreshed {path.name}: {vals[0][time_key]} → {vals[-1][time_key]}")


def main() -> None:
    """Shift timestamps in the dummy forecast inputs to end 7 days before today."""
    for name, time_key in DUMMY_PREDICT_FILES.items():
        refresh(DATA_DIR / name, time_key)


if __name__ == "__main__":
    main()
