import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_required_files_exist():
    for name in ["movies.csv", "movies_features.csv", "model.pkl", "scaler.pkl"]:
        assert (ROOT / name).exists()


def test_movies_csv_loads():
    df = pd.read_csv(ROOT / "movies.csv")
    assert len(df) > 0
    assert len(df.columns) > 1


def test_movies_features_csv_has_title_and_model_features():
    df = pd.read_csv(ROOT / "movies_features.csv")
    expected_columns = [
        "title", "budget", "popularity", "runtime", "vote_average",
        "vote_count", "genre_count", "release_year", "genre_Drama",
        "genre_Comedy", "genre_Thriller", "genre_Action", "genre_Romance",
        "genre_Adventure", "genre_Crime", "genre_Science", "genre_Fiction",
        "genre_Horror",
    ]

    assert df.columns.tolist() == expected_columns
    assert len(df) > 0
