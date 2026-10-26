
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "spotify_dataset_clean.csv"


def load_data(path=DATA_PATH):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {path}"
        )

    print("Membaca dataset bersih...")

    df = pd.read_csv(
        path,
        engine="c"
    )

    # Membersihkan nama kolom
    df.columns = (
        df.columns
        .str.strip()
        .str.strip('"')
        .str.strip("'")
    )

    print("Kolom asli setelah dibersihkan:")
    print(df.columns.tolist())

    rename_map = {
        "user_id": "user",
        "artistname": "artist",
        "trackname": "track",
        "playlistname": "playlist"
    }

    df = df.rename(columns=rename_map)

    required_columns = [
        "artist",
        "track",
        "playlist"
    ]

    missing_columns = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Kolom tidak ditemukan: {missing_columns}\n"
            f"Kolom tersedia: {list(df.columns)}"
        )

    df = df[required_columns]

    df = df.dropna()

    df = df.drop_duplicates()

    df = df.reset_index(drop=True)

    print("Dataset selesai dibaca.")
    print("Jumlah data setelah cleaning:", len(df))

    return df


def build_features(df):
    df = df.copy()

    track_counts = df["track"].value_counts()

    df["track_count"] = (
        df["track"]
        .map(track_counts)
        .astype(float)
    )

    X = df[["artist", "playlist"]]

    y = df["track_count"]

    return X, y, df
