
from pathlib import Path
import json


BASE_DIR = Path(__file__).resolve().parent

ARTIFACTS_DIR = (
    BASE_DIR /
    "artifacts"
)


def load_json(filename):

    path = (
        ARTIFACTS_DIR /
        filename
    )

    if not path.exists():
        return None

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def main():

    print("=" * 60)
    print("MODEL MONITORING")
    print("=" * 60)

    evaluation = load_json(
        "evaluation_results.json"
    )

    metadata = load_json(
        "model_metadata.json"
    )

    reference = load_json(
        "reference_stats.json"
    )

    print()
    print("EVALUATION")

    if evaluation:

        print(
            f"MSA / MAE : {evaluation['MSA_MAE']:.6f}"
        )

        print(
            f"MSE       : {evaluation['MSE']:.6f}"
        )

        print(
            f"R2        : {evaluation['R2']:.6f}"
        )

    else:

        print(
            "evaluation_results.json belum tersedia."
        )

    print()
    print("MODEL METADATA")

    if metadata:

        print(
            "Model type:",
            metadata["model_type"]
        )

        print(
            "Target:",
            metadata["target"]
        )

        print(
            "Training rows:",
            metadata["training_rows"]
        )

        print(
            "Testing rows:",
            metadata["testing_rows"]
        )

    else:

        print(
            "model_metadata.json belum tersedia."
        )

    print()
    print("REFERENCE STATISTICS")

    if reference:

        for key, value in reference.items():

            print(
                f"{key}: {value}"
            )

    else:

        print(
            "reference_stats.json belum tersedia."
        )

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
