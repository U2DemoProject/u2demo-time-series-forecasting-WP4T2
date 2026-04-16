"""Shift dummy_predict.json timestamps so the last value ends 7 days before today.

Run with:
    poetry run python scripts/refresh_dummy_predict.py
"""

import datetime as dt
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
TARGET_LAG_DAYS = 7


def main() -> None:
    """Shift timestamps in dummy_predict.json to end 7 days before today."""
    path = DATA_DIR / "dummy_predict.json"
    with open(path) as f:
        d = json.load(f)

    vals = d["target_series"]["historical_values"]
    last_time = dt.datetime.fromisoformat(vals[-1]["time"])
    target_end = dt.datetime.now() - dt.timedelta(days=TARGET_LAG_DAYS)
    shift = dt.timedelta(days=(target_end.date() - last_time.date()).days)

    for v in vals:
        v["time"] = (dt.datetime.fromisoformat(v["time"]) + shift).isoformat()

    with open(path, "w") as f:
        json.dump(d, f, indent=2)

    print(f"Refreshed: {vals[0]['time']} → {vals[-1]['time']}")


if __name__ == "__main__":
    main()
