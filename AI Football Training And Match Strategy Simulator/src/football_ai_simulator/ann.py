from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from .data import FEATURE_COLUMNS, TARGET_COLUMNS, MatchSituation, generate_football_dataset
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data import FEATURE_COLUMNS, TARGET_COLUMNS, MatchSituation, generate_football_dataset


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


MODEL_FILE = "football_ann.keras"
SCALER_FILE = "football_scaler.joblib"
METRICS_FILE = "training_metrics.json"
REPORT_FILE = "training_report.txt"
DATASET_FILE = "training_dataset.csv"


@dataclass(frozen=True)
class Prediction:
    pass_success_probability: float
    goal_probability: float
    fatigue_risk: float
    recommended_strategy: str


def build_model(input_features: int = 5) -> Any:
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Sequential

    model = Sequential(
        [
            Input(shape=(input_features,)),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(3, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def set_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass


def write_training_report(metrics: dict[str, float], model_dir: Path) -> None:
    report = f"""AI Football ANN Training Report

Training status: SUCCESSFULLY TRAINED

Dataset
- Total synthetic rows: {metrics['rows']:.0f}
- Training rows: {metrics['training_rows']:.0f}
- Test rows: {metrics['test_rows']:.0f}
- Pass success rows: {metrics['pass_success_rows']:.0f}
- Pass failure rows: {metrics['pass_failure_rows']:.0f}
- Input features: {", ".join(FEATURE_COLUMNS)}
- Prediction targets: {", ".join(TARGET_COLUMNS)}

Model
- Algorithm: Feedforward Artificial Neural Network
- Epochs requested: {metrics['epochs_requested']:.0f}
- Epochs trained: {metrics['epochs_trained']:.0f}

Evaluation
- Pass success accuracy: {metrics['pass_accuracy'] * 100:.2f}%
- Precision: {metrics['pass_precision'] * 100:.2f}%
- Recall: {metrics['pass_recall'] * 100:.2f}%
- Test MAE: {metrics['test_mae']:.4f}
- Test loss: {metrics['test_loss']:.4f}
- True positives: {metrics['true_positives']:.0f}
- True negatives: {metrics['true_negatives']:.0f}
- False positives: {metrics['false_positives']:.0f}
- False negatives: {metrics['false_negatives']:.0f}

Saved Files
- Model: {MODEL_FILE}
- Scaler: {SCALER_FILE}
- Metrics JSON: {METRICS_FILE}
- Training dataset CSV: data/{DATASET_FILE}

Explanation
The ANN was trained on synthetic football match situations. Each row contains stamina,
distance from goal, nearby defenders, passing accuracy, and match minute. The model
learns to predict pass success, goal probability, and fatigue risk.

The dataset is intentionally controlled for a semester project. Pass success is generated
from an explainable football rule: stamina and passing accuracy increase the chance, while
nearby defenders, long distance, and late match minutes reduce it. Because the labels follow
this clear rule, the ANN can learn the pattern with high accuracy.
"""
    (model_dir / REPORT_FILE).write_text(report, encoding="utf-8")


def recommend_strategy(
    pass_success_probability: float,
    goal_probability: float,
    fatigue_risk: float,
    nearby_defenders: float,
) -> str:
    if fatigue_risk >= 0.74:
        return "Slow tempo and recycle possession"
    if goal_probability >= 0.58:
        return "Shoot"
    if pass_success_probability >= 0.68 and nearby_defenders >= 3:
        return "Quick one-touch pass"
    if nearby_defenders >= 4:
        return "Switch play"
    if pass_success_probability >= 0.62:
        return "Forward pass"
    return "Carry ball into space"


def train_ann(
    rows: int = 1600,
    epochs: int = 20,
    seed: int = 7,
    model_dir: str | Path = "models",
    data_dir: str | Path = "data",
    verbose: int = 0,
) -> dict[str, float]:
    from tensorflow.keras.callbacks import EarlyStopping

    set_training_seed(seed)

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(data_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_football_dataset(rows=rows, seed=seed)
    dataset.to_csv(dataset_dir / DATASET_FILE, index=False)
    x = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMNS]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=dataset["pass_success"],
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = build_model(input_features=len(FEATURE_COLUMNS))
    history = model.fit(
        x_train_scaled,
        y_train.to_numpy(dtype=np.float32),
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        verbose=verbose,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=4,
                restore_best_weights=True,
            )
        ],
    )

    loss, mae = model.evaluate(x_test_scaled, y_test.to_numpy(dtype=np.float32), verbose=0)
    raw_predictions = model.predict(x_test_scaled, verbose=0)
    pass_predictions = (raw_predictions[:, 0] >= 0.5).astype(int)
    pass_labels = y_test["pass_success"].to_numpy()
    pass_accuracy = float(np.mean(pass_predictions == pass_labels))
    true_positives = int(np.sum((pass_predictions == 1) & (pass_labels == 1)))
    true_negatives = int(np.sum((pass_predictions == 0) & (pass_labels == 0)))
    false_positives = int(np.sum((pass_predictions == 1) & (pass_labels == 0)))
    false_negatives = int(np.sum((pass_predictions == 0) & (pass_labels == 1)))
    pass_precision = true_positives / max(1, true_positives + false_positives)
    pass_recall = true_positives / max(1, true_positives + false_negatives)

    model.save(model_dir / MODEL_FILE)
    joblib.dump(scaler, model_dir / SCALER_FILE)

    metrics = {
        "rows": float(rows),
        "training_rows": float(len(x_train)),
        "test_rows": float(len(x_test)),
        "pass_success_rows": float(dataset["pass_success"].sum()),
        "pass_failure_rows": float((dataset["pass_success"] == 0).sum()),
        "epochs_requested": float(epochs),
        "epochs_trained": float(len(history.history.get("loss", []))),
        "test_loss": float(loss),
        "test_mae": float(mae),
        "pass_accuracy": pass_accuracy,
        "pass_precision": float(pass_precision),
        "pass_recall": float(pass_recall),
        "true_positives": float(true_positives),
        "true_negatives": float(true_negatives),
        "false_positives": float(false_positives),
        "false_negatives": float(false_negatives),
    }
    (model_dir / METRICS_FILE).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_training_report(metrics, model_dir)
    return metrics


class MatchPredictor:
    def __init__(self, model: Any, scaler: StandardScaler, metrics: dict[str, float] | None = None):
        self.model = model
        self.scaler = scaler
        self.metrics = metrics or {}

    @classmethod
    def load(cls, model_dir: str | Path = "models") -> "MatchPredictor":
        from tensorflow.keras.models import load_model

        model_dir = Path(model_dir)
        model = load_model(model_dir / MODEL_FILE)
        scaler = joblib.load(model_dir / SCALER_FILE)
        metrics_path = model_dir / METRICS_FILE
        metrics = {}
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return cls(model=model, scaler=scaler, metrics=metrics)

    @classmethod
    def load_or_train(
        cls,
        model_dir: str | Path = "models",
        rows: int = 1600,
        epochs: int = 20,
        seed: int = 7,
    ) -> "MatchPredictor":
        model_dir = Path(model_dir)
        if not (model_dir / MODEL_FILE).exists() or not (model_dir / SCALER_FILE).exists():
            train_ann(rows=rows, epochs=epochs, seed=seed, model_dir=model_dir, verbose=0)
        return cls.load(model_dir=model_dir)

    def predict(self, situation: MatchSituation) -> Prediction:
        frame = situation.as_frame()
        scaled = self.scaler.transform(frame)
        pass_probability, goal_probability, fatigue_risk = self.model.predict(
            scaled,
            verbose=0,
        )[0]

        strategy = recommend_strategy(
            pass_success_probability=float(pass_probability),
            goal_probability=float(goal_probability),
            fatigue_risk=float(fatigue_risk),
            nearby_defenders=situation.nearby_defenders,
        )

        return Prediction(
            pass_success_probability=float(pass_probability),
            goal_probability=float(goal_probability),
            fatigue_risk=float(fatigue_risk),
            recommended_strategy=strategy,
        )

    def training_proof_text(self) -> str:
        if not self.metrics:
            return "ANN trained | metrics not found"

        accuracy = self.metrics.get("pass_accuracy")
        rows = self.metrics.get("rows")
        mae = self.metrics.get("test_mae")

        parts = ["ANN trained"]
        if accuracy is not None:
            parts.append(f"accuracy {accuracy * 100:.1f}%")
        if rows is not None:
            parts.append(f"data {rows:.0f} rows")
        if mae is not None:
            parts.append(f"MAE {mae:.3f}")
        return " | ".join(parts)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    metrics = train_ann(
        rows=1600,
        epochs=20,
        seed=7,
        model_dir=project_root / "models",
        data_dir=project_root / "data",
        verbose=1,
    )

    print("\nANN trained successfully")
    print(f"Pass success accuracy: {metrics['pass_accuracy'] * 100:.2f}%")
    print(f"Test MAE: {metrics['test_mae']:.4f}")
    print(f"Training report: {project_root / 'models' / REPORT_FILE}")
    print(f"Training dataset: {project_root / 'data' / DATASET_FILE}")


if __name__ == "__main__":
    main()
