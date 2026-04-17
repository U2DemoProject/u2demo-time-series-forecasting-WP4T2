from pathlib import Path

BASEDIR = Path(__file__).parents[2]
CACHE_DIR = BASEDIR / ".cache"
MODEL_DIR = BASEDIR / "models"
OUTPUT_DIR = BASEDIR / "output"

METADATA_FILENAME = "u2demo/metadata.json"
LEADERBOARD_FILENAME = "u2demo/training_result_leaderboard.csv"
