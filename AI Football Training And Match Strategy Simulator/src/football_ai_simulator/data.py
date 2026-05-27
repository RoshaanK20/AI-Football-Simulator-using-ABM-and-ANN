from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "stamina",
    "distance_goal",
    "nearby_defenders",
    "passing_accuracy",
    "match_minute",
]

TARGET_COLUMNS = ["pass_success", "goal_probability", "fatigue_risk"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "training_dataset.csv"


@dataclass(frozen=True)
class MatchSituation:
    stamina: float
    distance_goal: float
    nearby_defenders: float
    passing_accuracy: float
    match_minute: float

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [[
                self.stamina,
                self.distance_goal,
                self.nearby_defenders,
                self.passing_accuracy,
                self.match_minute,
            ]],
            columns=FEATURE_COLUMNS,
        )


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def generate_football_dataset(rows: int = 1600, seed: int = 7) -> pd.DataFrame:
    """Create a synthetic but football-shaped dataset for the ANN."""
    rng = np.random.default_rng(seed)

    stamina = rng.integers(35, 101, rows)
    distance_goal = rng.integers(1, 56, rows)
    nearby_defenders = rng.integers(0, 6, rows)
    passing_accuracy = rng.integers(45, 100, rows)
    match_minute = rng.integers(1, 91, rows)

    pass_score = (
        stamina * 0.36
        + passing_accuracy * 0.56
        - nearby_defenders * 8.7
        - distance_goal * 0.23
        - match_minute * 0.055
        + rng.normal(0, 1.5, rows)
    )
    pass_success = (pass_score >= 35).astype(int)

    goal_logit = (
        2.5
        + stamina * 0.015
        + passing_accuracy * 0.011
        - distance_goal * 0.11
        - nearby_defenders * 0.48
        - match_minute * 0.004
    )
    goal_probability = np.clip(sigmoid(goal_logit), 0.02, 0.96)

    fatigue_risk = np.clip(
        1.0
        - stamina / 100.0
        + match_minute / 132.0
        + nearby_defenders * 0.04
        + distance_goal / 500.0,
        0.02,
        0.98,
    )

    return pd.DataFrame(
        {
            "stamina": stamina,
            "distance_goal": distance_goal,
            "nearby_defenders": nearby_defenders,
            "passing_accuracy": passing_accuracy,
            "match_minute": match_minute,
            "pass_success": pass_success,
            "goal_probability": goal_probability,
            "fatigue_risk": fatigue_risk,
        }
    )


def save_generated_dataset(
    rows: int = 1600,
    seed: int = 7,
    output_path: str | Path = DEFAULT_DATASET_PATH,
) -> pd.DataFrame:
    dataset = generate_football_dataset(rows=rows, seed=seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    return dataset


def main() -> None:
    dataset = save_generated_dataset()
    print("Football dataset generated successfully")
    print(f"Rows: {len(dataset)}")
    print(f"Saved CSV: {DEFAULT_DATASET_PATH}")
    print("\nGeneration logic:")
    print("- Higher stamina and passing accuracy increase pass success.")
    print("- More defenders, longer distance from goal, and late match minutes reduce pass success.")
    print("- Goal probability and fatigue risk are calculated from football-style formulas.")
    print("\nSample rows:")
    print(dataset.head().to_string(index=False))


if __name__ == "__main__":
    main()
