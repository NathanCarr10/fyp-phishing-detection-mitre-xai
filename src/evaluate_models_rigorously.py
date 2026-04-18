"""
Rigorous evaluation for phishing detection models.

This script adds stronger academic evaluation beyond a single train/test split by using:
- Repeated Stratified K-Fold cross-validation
- Confidence intervals from fold score distributions
- Threshold sensitivity analysis
- Data-driven threshold recommendation
- Calibration checks (Brier score + ECE)

Run:
    python src/evaluate_models_rigorously.py
    python src/evaluate_models_rigorously.py --quick
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import MultinomialNB

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
PROJECT_ROOT = SRC_DIR.parent

BALANCED_DATA_PATH = "data/processed/english_dataset_balanced.csv"
DATA_PATH = BALANCED_DATA_PATH if os.path.exists(BALANCED_DATA_PATH) else "data/processed/english_dataset.csv"
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"
FOLD_METRICS_PATH = OUTPUT_DIR / "cv_fold_metrics.csv"
SUMMARY_METRICS_PATH = OUTPUT_DIR / "cv_summary_metrics.csv"
THRESHOLD_PATH = OUTPUT_DIR / "threshold_sensitivity_logreg.csv"
CALIBRATION_PATH = OUTPUT_DIR / "calibration_logreg.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rigorous model evaluation.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer folds/repeats for a faster debug run.",
    )
    return parser.parse_args()


def load_data(path: str, text_col: str, label_col: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path).dropna(subset=[text_col, label_col])
    texts = df[text_col].astype(str).values
    labels = df[label_col].astype(int).values
    return texts, labels


def create_model(model_name: str):
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    if model_name == "naive_bayes":
        return MultinomialNB()
    raise ValueError(f"Unknown model_name: {model_name}")


def classify_with_threshold(proba: np.ndarray, threshold: float) -> np.ndarray:
    return (proba >= threshold).astype(int)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) using equal-width bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1

    ece = 0.0
    n = len(y_true)

    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        if not np.any(mask):
            continue

        avg_conf = float(np.mean(y_prob[mask]))
        avg_acc = float(np.mean(y_true[mask]))
        bin_weight = float(np.sum(mask) / n)
        ece += abs(avg_acc - avg_conf) * bin_weight

    return float(ece)


def confidence_interval_from_samples(values: list[float]) -> tuple[float, float, float, float]:
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    ci_low = float(np.percentile(arr, 2.5))
    ci_high = float(np.percentile(arr, 97.5))
    return mean, std, ci_low, ci_high


def evaluate_model_cv(
    model_name: str,
    texts: np.ndarray,
    labels: np.ndarray,
    n_splits: int,
    n_repeats: int,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Evaluate one model with repeated stratified CV.

    Returns:
        fold_df: per-fold metrics
        all_true: concatenated out-of-fold true labels
        all_prob: concatenated out-of-fold phishing probabilities
    """
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    fold_rows: list[dict] = []
    all_true: list[np.ndarray] = []
    all_prob: list[np.ndarray] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(texts, labels), start=1):
        X_train_text = texts[train_idx]
        X_test_text = texts[test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=5000,
        )
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)

        model = create_model(model_name)
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)
        phishing_idx = list(model.classes_).index(1) if 1 in model.classes_ else 1
        y_prob = proba[:, phishing_idx]
        y_pred = classify_with_threshold(y_prob, threshold=0.5)

        fold_rows.append(
            {
                "model": model_name,
                "fold": fold_idx,
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "auc": float(roc_auc_score(y_test, y_prob)),
            }
        )

        all_true.append(y_test)
        all_prob.append(y_prob)

    fold_df = pd.DataFrame(fold_rows)
    return fold_df, np.concatenate(all_true), np.concatenate(all_prob)


def build_summary(fold_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict] = []

    for model_name, group in fold_df.groupby("model"):
        for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
            values = group[metric].tolist()
            mean, std, ci_low, ci_high = confidence_interval_from_samples(values)
            summary_rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "n_folds": len(values),
                }
            )

    return pd.DataFrame(summary_rows)


def threshold_sensitivity(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    rows: list[dict] = []

    for threshold in np.arange(0.10, 0.96, 0.05):
        y_pred = classify_with_threshold(y_prob, float(threshold))

        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        accuracy = float(accuracy_score(y_true, y_pred))

        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
            }
        )

    return pd.DataFrame(rows)


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1

    rows = []
    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        if not np.any(mask):
            rows.append(
                {
                    "bin": bin_idx,
                    "bin_start": bins[bin_idx],
                    "bin_end": bins[bin_idx + 1],
                    "count": 0,
                    "avg_confidence": np.nan,
                    "empirical_positive_rate": np.nan,
                }
            )
            continue

        rows.append(
            {
                "bin": bin_idx,
                "bin_start": bins[bin_idx],
                "bin_end": bins[bin_idx + 1],
                "count": int(np.sum(mask)),
                "avg_confidence": float(np.mean(y_prob[mask])),
                "empirical_positive_rate": float(np.mean(y_true[mask])),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    n_splits = 3 if args.quick else 5
    n_repeats = 1 if args.quick else 3

    print("=" * 70)
    print("RIGOROUS MODEL EVALUATION")
    print("=" * 70)
    print(f"Data path: {DATA_PATH}")
    print(f"CV setup: {n_splits} folds x {n_repeats} repeats")

    texts, labels = load_data(DATA_PATH, TEXT_COLUMN, LABEL_COLUMN)

    logreg_fold_df, logreg_true, logreg_prob = evaluate_model_cv(
        model_name="logistic_regression",
        texts=texts,
        labels=labels,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=42,
    )

    nb_fold_df, _, _ = evaluate_model_cv(
        model_name="naive_bayes",
        texts=texts,
        labels=labels,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=42,
    )

    fold_df = pd.concat([logreg_fold_df, nb_fold_df], ignore_index=True)
    summary_df = build_summary(fold_df)

    threshold_df = threshold_sensitivity(logreg_true, logreg_prob)
    best_idx = threshold_df["f1"].idxmax()
    best_threshold = float(threshold_df.loc[best_idx, "threshold"])

    brier = float(brier_score_loss(logreg_true, logreg_prob))
    ece = expected_calibration_error(logreg_true, logreg_prob, n_bins=10)

    calibration_df = calibration_table(logreg_true, logreg_prob, n_bins=10)
    calibration_df["brier_score"] = brier
    calibration_df["ece"] = ece

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(FOLD_METRICS_PATH, index=False)
    summary_df.to_csv(SUMMARY_METRICS_PATH, index=False)
    threshold_df.to_csv(THRESHOLD_PATH, index=False)
    calibration_df.to_csv(CALIBRATION_PATH, index=False)

    print("\nSaved outputs:")
    print(f"- {FOLD_METRICS_PATH}")
    print(f"- {SUMMARY_METRICS_PATH}")
    print(f"- {THRESHOLD_PATH}")
    print(f"- {CALIBRATION_PATH}")

    print("\nThreshold recommendation (LogReg):")
    print(f"- Best threshold by F1: {best_threshold:.2f}")

    print("\nCalibration (LogReg):")
    print(f"- Brier score: {brier:.6f}")
    print(f"- ECE (10 bins): {ece:.6f}")

    # Keep a small compatibility summary for quick reporting in thesis tables.
    compact = summary_df.pivot(index="metric", columns="model", values="mean").reset_index()
    compact = compact.rename(
        columns={
            "metric": "Metric",
            "logistic_regression": "Logistic Regression (CV Mean)",
            "naive_bayes": "Naive Bayes (CV Mean)",
        }
    )
    compact.to_csv(PROJECT_ROOT / "model_comparison.csv", index=False)
    print("\nUpdated model_comparison.csv with CV mean metrics.")


if __name__ == "__main__":
    main()
