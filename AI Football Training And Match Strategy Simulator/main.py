from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_ai_simulator.ann import train_ann


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI football simulator using ANN predictions and agent-based player behavior."
    )
    parser.add_argument("--train-only", action="store_true", help="Train the ANN and exit.")
    parser.add_argument("--rows", type=int, default=1600, help="Synthetic dataset rows to generate.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--model-dir", default="models", help="Directory for saved model artifacts.")
    parser.add_argument(
        "--match-seconds",
        type=float,
        default=120.0,
        help="Real seconds for a full 90-minute simulated match. Lower is faster.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = ROOT / args.model_dir

    if args.train_only:
        metrics = train_ann(
            rows=args.rows,
            epochs=args.epochs,
            seed=args.seed,
            model_dir=model_dir,
            data_dir=ROOT / "data",
            verbose=1,
        )
        print("\nTraining complete")
        print(f"Pass accuracy: {metrics['pass_accuracy']:.3f}")
        print(f"Precision: {metrics['pass_precision']:.3f}")
        print(f"Recall: {metrics['pass_recall']:.3f}")
        print(f"Test MAE: {metrics['test_mae']:.3f}")
        print(f"Saved model: {model_dir / 'football_ann.keras'}")
        print(f"Saved report: {model_dir / 'training_report.txt'}")
        print(f"Saved dataset: {ROOT / 'data' / 'training_dataset.csv'}")
        return

    from football_ai_simulator.simulation import run_simulation

    run_simulation(
        model_dir=model_dir,
        rows=args.rows,
        epochs=args.epochs,
        seed=args.seed,
        match_seconds=args.match_seconds,
    )


if __name__ == "__main__":
    main()
