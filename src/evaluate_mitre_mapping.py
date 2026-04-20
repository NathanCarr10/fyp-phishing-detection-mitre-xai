"""
Checks how well my MITRE mapping works on a manually labelled subset.

Validation dataset:
    data/processed/mitre_validation_subset.csv

Outputs:
    evaluation_results/mitre_mapping_predictions.csv
    evaluation_results/mitre_mapping_summary.csv

Run:
    python src/evaluate_mitre_mapping.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from attacker_sim import mitre_mapping

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
PROJECT_ROOT = SRC_DIR.parent

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "mitre_validation_subset.csv"
OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"
PREDICTIONS_PATH = OUTPUT_DIR / "mitre_mapping_predictions.csv"
SUMMARY_PATH = OUTPUT_DIR / "mitre_mapping_summary.csv"


def extract_technique_id(label: str) -> str:
    """Pull the technique code from the full label string."""
    return str(label).split(" - ")[0].strip()


def parse_expected_all(expected_all: str) -> set[str]:
    """Turn the semicolon-separated label list into a set."""
    parts = [p.strip() for p in str(expected_all).split(";") if p.strip()]
    return set(parts)


def evaluate_multi_label_micro(rows: list[dict], labels: list[str]) -> tuple[float, float, float]:
    """Compute micro precision, recall, and F1 for the set labels."""
    tp = fp = fn = 0

    for row in rows:
        true_set = row["expected_all_set"]
        pred_set = row["predicted_all_set"]

        tp += len(true_set & pred_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return float(precision), float(recall), float(f1)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Validation file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    required_cols = {"text", "expected_primary", "expected_all"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in validation file: {sorted(missing)}")

    rows: list[dict] = []

    for _, item in df.iterrows():
        text = str(item["text"])

        predicted_primary_full = mitre_mapping(text, return_all=False)
        predicted_primary = extract_technique_id(str(predicted_primary_full))

        predicted_all_full = mitre_mapping(text, return_all=True)
        if isinstance(predicted_all_full, list):
            predicted_all = [extract_technique_id(str(v)) for v in predicted_all_full]
        else:
            predicted_all = [extract_technique_id(str(predicted_all_full))]

        expected_primary = str(item["expected_primary"]).strip()
        expected_all_set = parse_expected_all(str(item["expected_all"]))
        predicted_all_set = set(predicted_all)

        rows.append(
            {
                "sample_id": item.get("sample_id", ""),
                "expected_primary": expected_primary,
                "predicted_primary": predicted_primary,
                "primary_correct": int(predicted_primary == expected_primary),
                "expected_all": ";".join(sorted(expected_all_set)),
                "predicted_all": ";".join(sorted(predicted_all_set)),
                "all_exact_match": int(expected_all_set == predicted_all_set),
                "expected_all_set": expected_all_set,
                "predicted_all_set": predicted_all_set,
            }
        )

    labels = sorted({r["expected_primary"] for r in rows} | {r["predicted_primary"] for r in rows})

    y_true = [r["expected_primary"] for r in rows]
    y_pred = [r["predicted_primary"] for r in rows]

    per_label_precision, per_label_recall, per_label_f1, per_label_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    accuracy = sum(r["primary_correct"] for r in rows) / len(rows) if rows else 0.0
    exact_match_rate = sum(r["all_exact_match"] for r in rows) / len(rows) if rows else 0.0

    micro_precision, micro_recall, micro_f1 = evaluate_multi_label_micro(rows, labels)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.DataFrame(
        [
            {
                "sample_id": r["sample_id"],
                "expected_primary": r["expected_primary"],
                "predicted_primary": r["predicted_primary"],
                "primary_correct": r["primary_correct"],
                "expected_all": r["expected_all"],
                "predicted_all": r["predicted_all"],
                "all_exact_match": r["all_exact_match"],
            }
            for r in rows
        ]
    )
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)

    summary_rows = [
        {"scope": "overall_primary", "metric": "accuracy", "value": float(accuracy)},
        {"scope": "overall_primary", "metric": "macro_precision", "value": float(macro_precision)},
        {"scope": "overall_primary", "metric": "macro_recall", "value": float(macro_recall)},
        {"scope": "overall_primary", "metric": "macro_f1", "value": float(macro_f1)},
        {"scope": "overall_multilabel", "metric": "exact_match_rate", "value": float(exact_match_rate)},
        {"scope": "overall_multilabel", "metric": "micro_precision", "value": float(micro_precision)},
        {"scope": "overall_multilabel", "metric": "micro_recall", "value": float(micro_recall)},
        {"scope": "overall_multilabel", "metric": "micro_f1", "value": float(micro_f1)},
    ]

    precision_arr = np.atleast_1d(per_label_precision)
    recall_arr = np.atleast_1d(per_label_recall)
    f1_arr = np.atleast_1d(per_label_f1)
    support_arr = np.atleast_1d(per_label_support if per_label_support is not None else np.array([]))

    for label, p, r, f, s in zip(labels, precision_arr, recall_arr, f1_arr, support_arr):
        summary_rows.append({"scope": f"primary_per_label:{label}", "metric": "precision", "value": float(p)})
        summary_rows.append({"scope": f"primary_per_label:{label}", "metric": "recall", "value": float(r)})
        summary_rows.append({"scope": f"primary_per_label:{label}", "metric": "f1", "value": float(f)})
        summary_rows.append({"scope": f"primary_per_label:{label}", "metric": "support", "value": float(s)})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("MITRE mapping evaluation complete.")
    print(f"Saved predictions to: {PREDICTIONS_PATH}")
    print(f"Saved summary to: {SUMMARY_PATH}")
    print("\nPrimary-label metrics:")
    print(f"- Accuracy: {accuracy:.4f}")
    print(f"- Macro Precision: {macro_precision:.4f}")
    print(f"- Macro Recall: {macro_recall:.4f}")
    print(f"- Macro F1: {macro_f1:.4f}")
    print("\nMulti-label metrics:")
    print(f"- Exact Match Rate: {exact_match_rate:.4f}")
    print(f"- Micro Precision: {micro_precision:.4f}")
    print(f"- Micro Recall: {micro_recall:.4f}")
    print(f"- Micro F1: {micro_f1:.4f}")


if __name__ == "__main__":
    main()
