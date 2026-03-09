# src/visualise_results.py
#
# Generate visualisations from attacker simulation analysis outputs.
#
# Inputs:
#   simulation_results/analysis/summary_by_threshold.csv
#   simulation_results/analysis/summary_by_attack_type.csv
#   simulation_results/analysis/summary_by_rule_chain.csv
#   simulation_results/analysis/summary_by_mitre.csv
#
# Outputs:
#   simulation_results/figures/threshold_detection_rate.png
#   simulation_results/figures/threshold_bypass_rate.png
#   simulation_results/figures/attack_type_detection_rate.png
#   simulation_results/figures/rule_chain_detection_rate.png
#   simulation_results/figures/mitre_detection_rate.png
#
# Notes:
# - Uses matplotlib only
# - One chart per figure
# - No custom colours/styles to keep plots simple and clean

import os
import pandas as pd
import matplotlib.pyplot as plt


ANALYSIS_DIR = os.path.join("simulation_results", "analysis")
FIGURES_DIR = os.path.join("simulation_results", "figures")

THRESHOLD_CSV = os.path.join(ANALYSIS_DIR, "summary_by_threshold.csv")
ATTACK_TYPE_CSV = os.path.join(ANALYSIS_DIR, "summary_by_attack_type.csv")
RULE_CHAIN_CSV = os.path.join(ANALYSIS_DIR, "summary_by_rule_chain.csv")
MITRE_CSV = os.path.join(ANALYSIS_DIR, "summary_by_mitre.csv")


def ensure_output_dir():
    """
    Create the figures output directory if it does not already exist.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)


def load_csv(path: str) -> pd.DataFrame | None:
    """
    Load a CSV file if it exists, otherwise return None.
    """
    if not os.path.exists(path):
        print(f"Missing file: {path}")
        return None
    return pd.read_csv(path)


def save_line_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_filename: str,
):
    """
    Save a simple line plot.
    """
    if df is None or df.empty:
        print(f"Skipping plot {output_filename}: no data.")
        return

    plot_df = df.copy().sort_values(by=x_col)

    plt.figure(figsize=(8, 5))
    plt.plot(plot_df[x_col], plot_df[y_col], marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()

    output_path = os.path.join(FIGURES_DIR, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def save_bar_plot(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_filename: str,
    sort_desc: bool = True,
    top_n: int | None = None,
    rotate_labels: bool = False,
):
    """
    Save a simple bar chart.
    """
    if df is None or df.empty:
        print(f"Skipping plot {output_filename}: no data.")
        return

    plot_df = df.copy()

    if sort_desc:
        plot_df = plot_df.sort_values(by=value_col, ascending=False)
    else:
        plot_df = plot_df.sort_values(by=value_col, ascending=True)

    if top_n is not None:
        plot_df = plot_df.head(top_n)

    plt.figure(figsize=(10, 6))
    plt.bar(plot_df[category_col].astype(str), plot_df[value_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if rotate_labels:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    output_path = os.path.join(FIGURES_DIR, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def create_threshold_plots():
    """
    Create threshold-based line plots.
    """
    df = load_csv(THRESHOLD_CSV)
    if df is None:
        return

    # Ensure numeric columns
    df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")
    df["detection_rate"] = pd.to_numeric(df["detection_rate"], errors="coerce")
    df["bypass_rate"] = pd.to_numeric(df["bypass_rate"], errors="coerce")
    df["precision"] = pd.to_numeric(df["precision"], errors="coerce")
    df["recall"] = pd.to_numeric(df["recall"], errors="coerce")
    df["f1"] = pd.to_numeric(df["f1"], errors="coerce")
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")

    save_line_plot(
        df=df,
        x_col="threshold",
        y_col="detection_rate",
        title="Detection Rate by Threshold",
        xlabel="Threshold",
        ylabel="Detection Rate",
        output_filename="threshold_detection_rate.png",
    )

    save_line_plot(
        df=df,
        x_col="threshold",
        y_col="bypass_rate",
        title="Bypass Rate by Threshold",
        xlabel="Threshold",
        ylabel="Bypass Rate",
        output_filename="threshold_bypass_rate.png",
    )

    save_line_plot(
        df=df,
        x_col="threshold",
        y_col="f1",
        title="F1 Score by Threshold",
        xlabel="Threshold",
        ylabel="F1 Score",
        output_filename="threshold_f1_score.png",
    )

    save_line_plot(
        df=df,
        x_col="threshold",
        y_col="accuracy",
        title="Accuracy by Threshold",
        xlabel="Threshold",
        ylabel="Accuracy",
        output_filename="threshold_accuracy.png",
    )


def create_attack_type_plot():
    """
    Create bar chart for detection rate by attack type.
    """
    df = load_csv(ATTACK_TYPE_CSV)
    if df is None:
        return

    df["detection_rate"] = pd.to_numeric(df["detection_rate"], errors="coerce")

    save_bar_plot(
        df=df,
        category_col="attack_type",
        value_col="detection_rate",
        title="Detection Rate by Attack Type",
        xlabel="Attack Type",
        ylabel="Detection Rate",
        output_filename="attack_type_detection_rate.png",
        sort_desc=False,
        rotate_labels=False,
    )


def create_rule_chain_plot():
    """
    Create bar chart for detection rate by rule chain.
    """
    df = load_csv(RULE_CHAIN_CSV)
    if df is None:
        return

    df["detection_rate"] = pd.to_numeric(df["detection_rate"], errors="coerce")

    # You may have many rule chains, so only plot top 15 for readability
    save_bar_plot(
        df=df,
        category_col="rule_chain",
        value_col="detection_rate",
        title="Detection Rate by Rule Chain",
        xlabel="Rule Chain",
        ylabel="Detection Rate",
        output_filename="rule_chain_detection_rate.png",
        sort_desc=False,
        top_n=15,
        rotate_labels=True,
    )


def create_mitre_plot():
    """
    Create bar chart for detection rate by MITRE technique.
    """
    df = load_csv(MITRE_CSV)
    if df is None:
        return

    df["detection_rate"] = pd.to_numeric(df["detection_rate"], errors="coerce")

    save_bar_plot(
        df=df,
        category_col="mitre_technique",
        value_col="detection_rate",
        title="Detection Rate by MITRE Technique",
        xlabel="MITRE Technique",
        ylabel="Detection Rate",
        output_filename="mitre_detection_rate.png",
        sort_desc=False,
        rotate_labels=True,
    )


def main():
    """
    Main visualisation runner.
    """
    ensure_output_dir()

    print("Generating visualisations from analysis CSV files...")
    create_threshold_plots()
    create_attack_type_plot()
    create_rule_chain_plot()
    create_mitre_plot()
    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()