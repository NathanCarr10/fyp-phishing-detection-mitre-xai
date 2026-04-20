"""
Generates error-analysis outputs for the phishing detector.

This script evaluates the saved Logistic Regression model on the holdout split
and writes CSV files that are used in the report discussion.

Run:
    python src/run_error_analysis.py
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
PROJECT_ROOT = SRC_DIR.parent

BALANCED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "english_dataset_balanced.csv"
FALLBACK_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "english_dataset.csv"

VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib"
MODEL_PATH = PROJECT_ROOT / "models" / "logreg_model.joblib"

OUTPUT_DIR = PROJECT_ROOT / "evaluation_results" / "error_analysis"

SEED = 42
TEST_SIZE = 0.2
TOP_TOKENS = 25
TOP_HARD_CASES = 50


def get_dataset_path() -> Path:
    if BALANCED_DATA_PATH.exists():
        return BALANCED_DATA_PATH
    return FALLBACK_DATA_PATH


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]{3,}", text.lower())


def top_tokens(texts: list[str], top_n: int) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))
    return counter.most_common(top_n)


def patch_legacy_logreg(clf) -> None:
    """Patch minimal attributes expected by newer sklearn versions."""
    if not hasattr(clf, "multi_class"):
        clf.multi_class = "auto"


def main() -> None:
    data_path = get_dataset_path()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    if not VECTORIZER_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts missing. Train first with: python src/mvp_baseline.py"
        )

    df = pd.read_csv(data_path)
    required_cols = {"text", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {sorted(required_cols)}")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    _, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["label"],
    )

    vectorizer = joblib.load(VECTORIZER_PATH)
    clf = joblib.load(MODEL_PATH)
    patch_legacy_logreg(clf)

    X_test = vectorizer.transform(test_df["text"])
    proba = clf.predict_proba(X_test)

    phishing_index = 1
    if 1 in clf.classes_:
        phishing_index = list(clf.classes_).index(1)

    test_df = test_df.reset_index(drop=True)
    test_df["phishing_probability"] = proba[:, phishing_index]
    test_df["pred_label"] = (test_df["phishing_probability"] >= 0.5).astype(int)
    test_df["pred_confidence"] = test_df["phishing_probability"].where(
        test_df["pred_label"] == 1,
        1.0 - test_df["phishing_probability"],
    )

    def error_type(row: pd.Series) -> str:
        if row["label"] == 1 and row["pred_label"] == 1:
            return "TP"
        if row["label"] == 0 and row["pred_label"] == 0:
            return "TN"
        if row["label"] == 0 and row["pred_label"] == 1:
            return "FP"
        return "FN"

    test_df["error_type"] = test_df.apply(error_type, axis=1)

    fp_df = test_df[test_df["error_type"] == "FP"].copy()
    fn_df = test_df[test_df["error_type"] == "FN"].copy()

    tn, fp, fn, tp = confusion_matrix(test_df["label"], test_df["pred_label"]).ravel()

    total = len(test_df)
    total_errors = len(fp_df) + len(fn_df)
    error_rate = total_errors / total if total else 0.0
    legit_total = int((test_df["label"] == 0).sum())
    phishing_total = int((test_df["label"] == 1).sum())

    summary_rows = [
        {"metric": "total_test_samples", "value": float(total)},
        {"metric": "total_errors", "value": float(total_errors)},
        {"metric": "error_rate", "value": float(error_rate)},
        {"metric": "false_positives", "value": float(fp)},
        {"metric": "false_negatives", "value": float(fn)},
        {"metric": "true_positives", "value": float(tp)},
        {"metric": "true_negatives", "value": float(tn)},
        {
            "metric": "false_positive_rate_within_legitimate",
            "value": float(fp / legit_total if legit_total else 0.0),
        },
        {
            "metric": "false_negative_rate_within_phishing",
            "value": float(fn / phishing_total if phishing_total else 0.0),
        },
    ]
    summary_df = pd.DataFrame(summary_rows)

    token_rows = []
    for token, count in top_tokens(fp_df["text"].tolist(), TOP_TOKENS):
        token_rows.append({"error_type": "FP", "token": token, "count": int(count)})
    for token, count in top_tokens(fn_df["text"].tolist(), TOP_TOKENS):
        token_rows.append({"error_type": "FN", "token": token, "count": int(count)})
    token_df = pd.DataFrame(token_rows)

    fp_hard_df = fp_df.sort_values("phishing_probability", ascending=False).head(TOP_HARD_CASES)
    fn_hard_df = fn_df.sort_values("phishing_probability", ascending=True).head(TOP_HARD_CASES)

    columns = ["text", "label", "pred_label", "phishing_probability", "pred_confidence", "error_type"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_DIR / "error_summary.csv", index=False)
    test_df[columns].to_csv(OUTPUT_DIR / "all_test_predictions.csv", index=False)
    fp_df[columns].to_csv(OUTPUT_DIR / "false_positives.csv", index=False)
    fn_df[columns].to_csv(OUTPUT_DIR / "false_negatives.csv", index=False)
    fp_hard_df[columns].to_csv(OUTPUT_DIR / "hard_false_positives.csv", index=False)
    fn_hard_df[columns].to_csv(OUTPUT_DIR / "hard_false_negatives.csv", index=False)
    token_df.to_csv(OUTPUT_DIR / "top_error_tokens.csv", index=False)

    report_lines = [
        "# Error Analysis Report",
        "",
        f"- Test samples: {total}",
        f"- Total errors: {total_errors}",
        f"- Error rate: {error_rate:.4f}",
        f"- False positives: {fp}",
        f"- False negatives: {fn}",
        "",
        "## Output Files",
        "- error_summary.csv",
        "- all_test_predictions.csv",
        "- false_positives.csv",
        "- false_negatives.csv",
        "- hard_false_positives.csv",
        "- hard_false_negatives.csv",
        "- top_error_tokens.csv",
    ]
    (OUTPUT_DIR / "error_analysis_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("Error analysis complete.")
    print(f"Saved outputs to: {OUTPUT_DIR}")
    print(f"- total errors: {total_errors}")
    print(f"- false positives: {fp}")
    print(f"- false negatives: {fn}")


if __name__ == "__main__":
    main()
