import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_required_files_exist():
    for name in ["movies.csv", "model.pkl", "scaler.pkl"]:
        assert (ROOT / name).exists()


def test_movies_csv_loads():
    df = pd.read_csv(ROOT / "movies.csv")
    assert len(df) > 0
    assert len(df.columns) > 1
