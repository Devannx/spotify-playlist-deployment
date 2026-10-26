
from pathlib import Path

import joblib
import pandas as pd

from flask import (
    Flask,
    jsonify,
    request
)


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR /
    "artifacts" /
    "spotify_playlist_model.joblib"
)


app = Flask(__name__)


def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Model belum tersedia."
        )

    return joblib.load(
        MODEL_PATH
    )


@app.get("/")
def home():

    return jsonify(
        {
            "project":
                "Spotify Playlist Model API",
            "status":
                "running",
            "endpoint":
                "POST /predict"
        }
    )


@app.get("/health")
def health():

    return jsonify(
        {
            "status":
                "healthy"
        }
    )


@app.post("/predict")
def predict():

    data = request.get_json(
        silent=True
    )

    if not data:

        return jsonify(
            {
                "error":
                    "JSON tidak ditemukan."
            }
        ), 400

    artist = data.get(
        "artist"
    )

    playlist = data.get(
        "playlist"
    )

    if not artist or not playlist:

        return jsonify(
            {
                "error":
                    "artist dan playlist wajib diisi."
            }
        ), 400

    try:

        model = load_model()

        input_data = pd.DataFrame(
            [
                {
                    "artist":
                        str(artist),
                    "playlist":
                        str(playlist)
                }
            ]
        )

        prediction = model.predict(
            input_data
        )[0]

        return jsonify(
            {
                "artist":
                    artist,
                "playlist":
                    playlist,
                "predicted_track_count":
                    round(
                        float(prediction),
                        4
                    )
            }
        )

    except Exception as error:

        return jsonify(
            {
                "error":
                    str(error)
            }
        ), 500


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
