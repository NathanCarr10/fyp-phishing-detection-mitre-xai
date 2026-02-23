# src/attacker_sim.py
#
# Upgraded attacker vs defender simulation.
#
# - Loads the saved model and vectorizer from mvp_baseline.py
# - Starts from simple phishing-style base messages
# - Applies simple "attack rules" (urgency, spoof bank, spoof Revenue, fake link)
# - Runs multiple rounds of simulations
# - Supports different probability thresholds for classifying phishing
# - Logs all results to a single CSV file with threshold and round info
# - Prints a summary detection/bypass rate per threshold
# - Now also reports recall and miss_rate for each threshold

import csv
import os
from datetime import datetime, timezone

from mvp_baseline import load_model  # uses your saved model


# Output CSV for logging results
SIM_OUTPUT_DIR = "simulation_results"
SIM_OUTPUT_PATH = os.path.join(SIM_OUTPUT_DIR, "attacker_simulation_log.csv")


# Simple MITRE mapping helper

def mitre_mapping(email_text: str) -> str:
    """
    Very simple mapping of phishing emails to MITRE ATT&CK T1566 sub-techniques.
    This is just a placeholder for now.
    - If it contains a link-like string → T1566.002 (Link)
    - Else → T1566.001 (Attachment/Generic)
    """
    text_lower = email_text.lower()

    if "http://" in text_lower or "https://" in text_lower or "www." in text_lower or "click here" in text_lower:
        return "T1566.002 - Phishing: Link"
    else:
        return "T1566.001 - Phishing: Attachment/Generic"


# Attacker rules

def add_urgency(text: str) -> str:
    return (
        "URGENT: Your account may be closed soon. " +
        text +
        " Please respond within 24 hours to avoid losing access."
    )


def spoof_bank(text: str) -> str:
    return (
        "AIB Security Notice: We detected unusual activity on your account. " +
        text +
        " Log in now to confirm your identity."
    )


def spoof_revenue(text: str) -> str:
    return (
        "Irish Revenue: You are eligible for a tax refund. " +
        text +
        " Please follow the link below to claim your refund."
    )


def add_fake_link(text: str) -> str:
    return (
        text +
        " Click here to resolve this issue: http://secure-verification-example.com/login"
    )


ATTACK_RULES = {
    "urgency": add_urgency,
    "spoof_bank": spoof_bank,
    "spoof_revenue": spoof_revenue,
    "fake_link": add_fake_link,
}


# Base phishing messages

BASE_PHISHING_EMAILS = [
    "We have detected suspicious activity on your account.",
    "Your password has expired and must be reset immediately.",
    "There is a problem with your billing information.",
    "We were unable to deliver your package due to an address issue.",
]


# Helper: classify with a custom threshold

def classify_email(vectorizer, clf, text: str, threshold: float = 0.5):
    """
    Classify a single email using a custom probability threshold.

    - Transforms text with the TF-IDF vectorizer
    - Uses clf.predict_proba to get phishing probability
    - If P(phishing) >= threshold → classify as phishing (1)
    - Else → classify as legit (0)
    """
    X = vectorizer.transform([text])
    proba = clf.predict_proba(X)[0]

    # Find index of the "phishing" class (label 1)
    if 1 in clf.classes_:
        phishing_index = list(clf.classes_).index(1)
    else:
        # fallback: assume second column is "positive"
        phishing_index = 1

    phishing_prob = float(proba[phishing_index])

    # Apply decision threshold
    pred_label = 1 if phishing_prob >= threshold else 0

    return pred_label, phishing_prob


# Core simulation for a single threshold

def run_simulation_for_threshold(
    vectorizer,
    clf,
    threshold: float,
    writer,
    num_rounds: int = 5,
    num_variants_per_base: int = 2,
):
    """
    Run the attacker vs defender simulation for one specific threshold.

    For each threshold:
      - Repeat num_rounds times to reduce randomness (even though rules are simple)
      - For each base email and each rule, create num_variants_per_base variants
      - Classify each variant using classify_email(..., threshold=threshold)
      - Log each attempt to CSV
      - Count attacker wins (phishing predicted as legit) and defender wins

    Returns a dict with summary stats for this threshold.
    """
    total_attacks = 0
    attacker_wins = 0
    defender_wins = 0

    for round_idx in range(num_rounds):
        for base_index, base_text in enumerate(BASE_PHISHING_EMAILS):
            for rule_name, rule_fn in ATTACK_RULES.items():
                for variant_idx in range(num_variants_per_base):
                    total_attacks += 1

                    modified_text = rule_fn(base_text)
                    mitre_label = mitre_mapping(modified_text)

                    # Classify with the given threshold
                    pred, phishing_prob = classify_email(
                        vectorizer, clf, modified_text, threshold=threshold
                    )

                    # Attacker "wins" if the model says this is legit (0)
                    attacker_success = int(pred == 0)
                    if attacker_success:
                        attacker_wins += 1
                    else:
                        defender_wins += 1

                    row = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "round": round_idx,
                        "threshold": threshold,
                        "base_index": base_index,
                        "base_text": base_text,
                        "rule_name": rule_name,
                        "variant_index": variant_idx,
                        "modified_text": modified_text,
                        "predicted_label": pred,
                        "phishing_probability": phishing_prob,
                        "attacker_success": attacker_success,
                        "mitre_technique": mitre_label,
                    }
                    writer.writerow(row)

    # Build summary for this threshold
    # In this simulation, all samples are phishing, so:
    # - defender_wins ≈ true positives
    # - attacker_wins ≈ false negatives
    tp = defender_wins
    fn = attacker_wins

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    miss_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    summary = {
        "threshold": threshold,
        "total_attacks": total_attacks,
        "attacker_wins": attacker_wins,
        "defender_wins": defender_wins,
        "detection_rate": defender_wins / total_attacks if total_attacks else 0.0,
        "bypass_rate": attacker_wins / total_attacks if total_attacks else 0.0,
        "recall": recall,
        "miss_rate": miss_rate,
    }
    return summary


# Top-level experiment runner

def run_experiments(
    thresholds=None,
    num_rounds: int = 5,
    num_variants_per_base: int = 2,
):
    """
    Run simulations for multiple thresholds and print a summary.

    thresholds: list of probability thresholds to test, e.g. [0.5, 0.6, 0.7]
    """
    if thresholds is None:
        thresholds = [0.5, 0.6, 0.7]

    # Load the saved model and vectorizer
    try:
        vectorizer, clf = load_model()
    except FileNotFoundError as e:
        print("\nModel files not found. Please run mvp_baseline.py once to train and save the model.")
        print("Error:", e)
        return

    os.makedirs(SIM_OUTPUT_DIR, exist_ok=True)

    fieldnames = [
        "timestamp",
        "round",
        "threshold",
        "base_index",
        "base_text",
        "rule_name",
        "variant_index",
        "modified_text",
        "predicted_label",
        "phishing_probability",
        "attacker_success",
        "mitre_technique",
    ]

    summaries = []

    # Open the CSV once, log all thresholds and rounds
    with open(SIM_OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for thr in thresholds:
            print(f"\nRunning simulation for threshold = {thr:.2f} ...")
            summary = run_simulation_for_threshold(
                vectorizer,
                clf,
                threshold=thr,
                writer=writer,
                num_rounds=num_rounds,
                num_variants_per_base=num_variants_per_base,
            )
            summaries.append(summary)

    # Print summary table
    print("\n=== Simulation Summary by Threshold ===")
    print(
        "Threshold | Total Attacks | Defender Wins | Attacker Wins | "
        "Detection Rate | Bypass Rate | Recall | Miss Rate"
    )
    for s in summaries:
        print(
            f"{s['threshold']:8.2f} | "
            f"{s['total_attacks']:13d} | "
            f"{s['defender_wins']:13d} | "
            f"{s['attacker_wins']:13d} | "
            f"{s['detection_rate']:14.3f} | "
            f"{s['bypass_rate']:11.3f} | "
            f"{s['recall']:6.3f} | "
            f"{s['miss_rate']:9.3f}"
        )

    print(f"\nDetailed log saved to: {SIM_OUTPUT_PATH}")


if __name__ == "__main__":
    # tweak these for experiments
    run_experiments(
        thresholds=[0.5, 0.6, 0.7],
        num_rounds=5,
        num_variants_per_base=2,
    )