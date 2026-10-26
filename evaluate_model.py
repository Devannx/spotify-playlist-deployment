
from pathlib import Path
import json
import joblib

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from spotify_features import (
    load_data,
    build_features
)


BASE_DIR = Path(__file__).resolve().parent

ARTIFACTS_DIR = (
    BASE_DIR /
    "artifacts"
)

MODEL_PATH = (
    ARTIFACTS_DIR /
    "spotify_playlist_model.joblib"
)


def main():

    print("=" * 60)
    print("EVALUASI MODEL")
    print("=" * 60)

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Model belum ditemukan. "
            "Jalankan train_model.py terlebih dahulu."
        )

    # Load dataset
    df = load_data()

    # Features dan target
    X, y, _ = build_features(df)

    # Split sama seperti training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Load model
    model = joblib.load(
        MODEL_PATH
    )

    # Prediction
    y_pred = model.predict(
        X_test
    )

    # ==================================================
    # 1. MSA / MAE
    # ==================================================

    msa = mean_absolute_error(
        y_test,
        y_pred
    )

    # ==================================================
    # 2. MSE
    # ==================================================

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    # ==================================================
    # 3. R2
    # ==================================================

    r2 = r2_score(
        y_test,
        y_pred
    )

    # Results
    results = {
        "MSA_MAE": float(msa),
        "MSE": float(mse),
        "R2": float(r2)
    }

    result_path = (
        ARTIFACTS_DIR /
        "evaluation_results.json"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

    # ==================================================
    # HASIL
    # ==================================================

    print()
    print("HASIL EVALUASI")
    print("-" * 60)

    print(
        f"MSA / MAE : {msa:.6f}"
    )

    print(
        f"MSE       : {mse:.6f}"
    )

    print(
        f"R2        : {r2:.6f}"
    )

    print("-" * 60)

    # ==================================================
    # PENJELASAN MSA / MAE
    # ==================================================

    print()
    print("PENJELASAN MSA / MAE")
    print("-" * 60)

    print(
        "MSA / MAE merupakan rata-rata "
        "nilai absolut dari kesalahan prediksi."
    )

    print(
        "Semakin kecil nilai MSA / MAE, "
        "semakin kecil kesalahan prediksi model."
    )

    # ==================================================
    # PENJELASAN MSE
    # ==================================================

    print()
    print("PENJELASAN MSE")
    print("-" * 60)

    print(
        "MSE merupakan rata-rata kuadrat "
        "dari kesalahan prediksi."
    )

    print(
        "Semakin kecil nilai MSE, "
        "semakin baik performa model."
    )

    print(
        "MSE lebih sensitif terhadap "
        "kesalahan prediksi yang besar."
    )

    # ==================================================
    # PENJELASAN R2
    # ==================================================

    print()
    print("PENJELASAN R2")
    print("-" * 60)

    print(
        "R2 menunjukkan seberapa besar "
        "variasi target yang dapat dijelaskan model."
    )

    print(
        "Nilai R2 yang semakin mendekati 1 "
        "menunjukkan model semakin baik."
    )

    print(
        "R2 = 0 berarti model setara "
        "dengan baseline rata-rata."
    )

    print(
        "R2 negatif berarti model lebih buruk "
        "daripada baseline rata-rata."
    )

    print()
    print(
        f"Hasil evaluasi disimpan di: {result_path}"
    )

    print()
    print("=" * 60)
    print("EVALUASI SELESAI")
    print("=" * 60)


if __name__ == "__main__":
    main()
