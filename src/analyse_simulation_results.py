# src/analyse_simulation_results.py
#
# Reads the attacker simulation log and creates summary tables.
#
# Input:
#   simulation_results/attacker_simulation_log.csv
#
# Output:
#   simulation_results/analysis/summary_by_threshold.csv
#   simulation_results/analysis/summary_by_attack_type.csv
#   simulation_results/analysis/summary_by_rule_chain.csv
#   simulation_results/analysis/summary_by_mitre.csv
#
# I use these outputs for writing up and plotting final results.

import os
import pandas as pd


INPUT_PATH = os.path.join("simulation_results", "attacker_simulation_log.csv")
OUTPUT_DIR = os.path.join("simulation_results", "analysis")


def safe_divide(numerator: float, denominator: float) -> float:
    """
    Safely divide two numbers, returning 0.0 if denominator is 0.
    """
    return numerator / denominator if denominator != 0 else 0.0


def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Compute confusion matrix and evaluation metrics for a dataframe subset.

    Expected columns:
    - true_label (1 = phishing, 0 = legit)
    - predicted_label (1 = predicted phishing, 0 = predicted legit)

    Returns a dictionary of metrics.
    """
    tp = int(((df["true_label"] == 1) & (df["predicted_label"] == 1)).sum())
    fp = int(((df["true_label"] == 0) & (df["predicted_label"] == 1)).sum())
    tn = int(((df["true_label"] == 0) & (df["predicted_label"] == 0)).sum())
    fn = int(((df["true_label"] == 1) & (df["predicted_label"] == 0)).sum())

    total_samples = tp + fp + tn + fn
    total_phishing = tp + fn

    detection_rate = safe_divide(tp, total_phishing)
    bypass_rate = safe_divide(fn, total_phishing)

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = safe_divide(tp + tn, total_samples)

    return {
        "total_samples": total_samples,
        "total_phishing": total_phishing,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "detection_rate": detection_rate,
        "bypass_rate": bypass_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def summarise_by_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """
    Group the simulation log by one or more columns and compute metrics for each group.

    Parameters
    ----------
    df : pd.DataFrame
        Full simulation log dataframe.
    group_cols : list[str]
        Columns to group by.

    Returns
    -------
    pd.DataFrame
        Summary dataframe with one row per group.
    """
    rows = []

    grouped = df.groupby(group_cols, dropna=False)

    for group_key, group_df in grouped:
        metrics = compute_metrics(group_df)

        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        row = {}
        for col_name, value in zip(group_cols, group_key):
            row[col_name] = value

        row.update(metrics)
        rows.append(row)

    summary_df = pd.DataFrame(rows)

    if not summary_df.empty:
        sort_cols = [col for col in group_cols if col in summary_df.columns]
        summary_df = summary_df.sort_values(sort_cols).reset_index(drop=True)

    return summary_df


def print_preview(title: str, df: pd.DataFrame, max_rows: int = 10):
    """
    Print a small preview of a summary dataframe to the console.
    """
    print(f"\n=== {title} ===")
    if df.empty:
        print("No rows found.")
        return
    print(df.head(max_rows).to_string(index=False))


def main():
    """
    Main analysis runner.
    """
    if not os.path.exists(INPUT_PATH):
        print(f"Input file not found: {INPUT_PATH}")
        print("Please run attacker_sim.py first.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading simulation log from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    required_columns = [
        "threshold",
        "attack_type",
        "rule_chain",
        "mitre_technique",
        "true_label",
        "predicted_label",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print("Missing required columns in simulation log:")
        for col in missing:
            print(f" - {col}")
        return

    # Ensure numeric columns are numeric
    df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")
    df["true_label"] = pd.to_numeric(df["true_label"], errors="coerce")
    df["predicted_label"] = pd.to_numeric(df["predicted_label"], errors="coerce")

    # --- Generate summaries ---

    summary_by_threshold = summarise_by_group(df, ["threshold"])
    summary_by_attack_type = summarise_by_group(df, ["attack_type"])
    summary_by_rule_chain = summarise_by_group(df, ["rule_chain"])
    summary_by_mitre = summarise_by_group(df, ["mitre_technique"])

    # Optional richer summaries
    summary_by_threshold_attack_type = summarise_by_group(df, ["threshold", "attack_type"])
    summary_by_threshold_rule_chain = summarise_by_group(df, ["threshold", "rule_chain"])
    summary_by_threshold_mitre = summarise_by_group(df, ["threshold", "mitre_technique"])

    # --- Save outputs ---

    out_threshold = os.path.join(OUTPUT_DIR, "summary_by_threshold.csv")
    out_attack_type = os.path.join(OUTPUT_DIR, "summary_by_attack_type.csv")
    out_rule_chain = os.path.join(OUTPUT_DIR, "summary_by_rule_chain.csv")
    out_mitre = os.path.join(OUTPUT_DIR, "summary_by_mitre.csv")
    out_threshold_attack_type = os.path.join(OUTPUT_DIR, "summary_by_threshold_attack_type.csv")
    out_threshold_rule_chain = os.path.join(OUTPUT_DIR, "summary_by_threshold_rule_chain.csv")
    out_threshold_mitre = os.path.join(OUTPUT_DIR, "summary_by_threshold_mitre.csv")

    summary_by_threshold.to_csv(out_threshold, index=False)
    summary_by_attack_type.to_csv(out_attack_type, index=False)
    summary_by_rule_chain.to_csv(out_rule_chain, index=False)
    summary_by_mitre.to_csv(out_mitre, index=False)
    summary_by_threshold_attack_type.to_csv(out_threshold_attack_type, index=False)
    summary_by_threshold_rule_chain.to_csv(out_threshold_rule_chain, index=False)
    summary_by_threshold_mitre.to_csv(out_threshold_mitre, index=False)

    # --- Console previews ---

    print_preview("Summary by Threshold", summary_by_threshold)
    print_preview("Summary by Attack Type", summary_by_attack_type)
    print_preview("Summary by Rule Chain", summary_by_rule_chain)
    print_preview("Summary by MITRE Technique", summary_by_mitre)

    print("\nAnalysis complete. Files saved to:")
    print(f" - {out_threshold}")
    print(f" - {out_attack_type}")
    print(f" - {out_rule_chain}")
    print(f" - {out_mitre}")
    print(f" - {out_threshold_attack_type}")
    print(f" - {out_threshold_rule_chain}")
    print(f" - {out_threshold_mitre}")


if __name__ == "__main__":
    main()