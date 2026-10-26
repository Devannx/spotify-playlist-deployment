
from pathlib import Path
import json
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from spotify_features import (
    load_data,
    build_features
)


BASE_DIR = Path(__file__).resolve().parent

ARTIFACTS_DIR = (
    BASE_DIR /
    "artifacts"
)

ARTIFACTS_DIR.mkdir(
    exist_ok=True
)


def main():

    print("=" * 60)
    print("TRAINING MODEL SPOTIFY PLAYLIST")
    print("=" * 60)

    # Load dataset
    df = load_data()

    print()
    print("Dataset berhasil dimuat.")
    print("Jumlah data:", len(df))

    # Feature engineering
    X, y, prepared_df = build_features(df)

    print()
    print("Feature engineering selesai.")
    print("Jumlah fitur:", X.shape[1])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print()
    print("Pembagian data:")
    print("Data training:", len(X_train))
    print("Data testing :", len(X_test))

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2
                ),
                [
                    "artist",
                    "playlist"
                ]
            )
        ]
    )

    # Model
    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "regressor",
                LinearRegression()
            )
        ]
    )

    # Training
    print()
    print("Memulai training...")

    model.fit(
        X_train,
        y_train
    )

    print("Training selesai.")

    # Save model
    model_path = (
        ARTIFACTS_DIR /
        "spotify_playlist_model.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    print()
    print("Model disimpan:")
    print(model_path)

    # Metadata
    metadata = {
        "model_type": "LinearRegression",
        "target": "track_count",
        "features": [
            "artist",
            "playlist"
        ],
        "test_size": 0.20,
        "random_state": 42,
        "total_rows": int(
            len(prepared_df)
        ),
        "training_rows": int(
            len(X_train)
        ),
        "testing_rows": int(
            len(X_test)
        )
    }

    metadata_path = (
        ARTIFACTS_DIR /
        "model_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Reference statistics
    reference_stats = {
        "target_mean": float(
            y.mean()
        ),
        "target_median": float(
            y.median()
        ),
        "target_min": float(
            y.min()
        ),
        "target_max": float(
            y.max()
        ),
        "unique_artists": int(
            df["artist"].nunique()
        ),
        "unique_tracks": int(
            df["track"].nunique()
        ),
        "unique_playlists": int(
            df["playlist"].nunique()
        )
    }

    reference_path = (
        ARTIFACTS_DIR /
        "reference_stats.json"
    )

    with open(
        reference_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            reference_stats,
            file,
            indent=4,
            ensure_ascii=False
        )

    print()
    print("Metadata berhasil disimpan.")
    print("Reference statistics berhasil disimpan.")

    print()
    print("=" * 60)
    print("TRAINING SELESAI")
    print("=" * 60)


if __name__ == "__main__":
    main()
