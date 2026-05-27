from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    from .ann import DATASET_FILE, REPORT_FILE, train_ann
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ann import DATASET_FILE, REPORT_FILE, train_ann


def main() -> None:
    parser = argparse.ArgumentParser(description="Live demo trainer for the football ANN predictor.")
    parser.add_argument("--rows", type=int, default=1600)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-dir", default=str(PROJECT_ROOT / "models"))
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    args = parser.parse_args()

    print("AI Football ANN Live Training Demo")
    print("1. Generating synthetic football match data")
    print("2. Splitting data into training and testing sets")
    print("3. Training the Feedforward Neural Network")
    print("4. Saving accuracy proof and trained model\n")

    metrics = train_ann(
        rows=args.rows,
        epochs=args.epochs,
        seed=args.seed,
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        verbose=1,
    )

    report_path = Path(args.model_dir) / REPORT_FILE
    dataset_path = Path(args.data_dir) / DATASET_FILE

    print("\nANN training completed successfully")
    print(f"Pass success accuracy: {metrics['pass_accuracy'] * 100:.2f}%")
    print(f"Precision: {metrics['pass_precision'] * 100:.2f}%")
    print(f"Recall: {metrics['pass_recall'] * 100:.2f}%")
    print(f"Test MAE: {metrics['test_mae']:.4f}")
    print(f"Training rows: {metrics['training_rows']:.0f}")
    print(f"Test rows: {metrics['test_rows']:.0f}")
    print(f"Training report: {report_path}")
    print(f"Training dataset: {dataset_path}")
    print("\nSir, this proves the model was trained and evaluated on generated football data.")


if __name__ == "__main__":
    main()
