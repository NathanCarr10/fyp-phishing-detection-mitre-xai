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
# - Prints a summary per threshold:
#     * Attacker vs defender wins (for phishing emails)
#     * Detection / bypass rate (for phishing emails)
#     * Full confusion matrix (TP, FP, TN, FN)
#     * Precision, recall, F1, accuracy (on phishing + legit)

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

    # Very naive link detection
    if "http://" in text_lower or "https://" in text_lower or "www." in text_lower or "click here" in text_lower:
        return "T1566.002 - Phishing: Link"
    else:
        # You could refine this later based on keywords like "attachment", "invoice.pdf", etc.
        return "T1566.001 - Phishing: Attachment/Generic"


# Attacker rules (used only for phishing emails)

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


# Base legitimate (non-phishing) emails

BASE_LEGIT_EMAILS = [
    "Hi, just a reminder that our team meeting is at 10am tomorrow.",
    "Please find attached the minutes from last week's project meeting.",
    "Thanks for your help with the report, I really appreciate it.",
    "Here is the updated timetable for the upcoming semester.",
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
      - Repeat num_rounds times
      - For each base phishing email and each rule, create num_variants_per_base variants
      - Classify each phishing variant (attacker vs defender)
      - Also classify a set of legitimate emails (no attacker)
      - Log each attempt to CSV
      - Track full confusion matrix and attacker/defender wins

    Returns a dict with summary stats for this threshold.
    """

    # Confusion matrix counters
    tp = fp = tn = fn = 0

    # Attacker vs defender counts (phishing only)
    total_phishing_attacks = 0
    attacker_wins = 0  # phishing slips through (false negative)
    defender_wins = 0  # phishing blocked (true positive)

    for round_idx in range(num_rounds):

        # 1) Phishing emails with attack rules
        for base_index, base_text in enumerate(BASE_PHISHING_EMAILS):
            for rule_name, rule_fn in ATTACK_RULES.items():
                for variant_idx in range(num_variants_per_base):
                    total_phishing_attacks += 1

                    modified_text = rule_fn(base_text)
                    mitre_label = mitre_mapping(modified_text)

                    # Classify with the given threshold
                    pred, phishing_prob = classify_email(
                        vectorizer, clf, modified_text, threshold=threshold
                    )

                    true_label = 1  # phishing
                    attacker_success = 0

                    if pred == 1:
                        # Correctly caught phishing
                        tp += 1
                        defender_wins += 1
                        attacker_success = 0
                    else:
                        # Missed phishing (attacker wins)
                        fn += 1
                        attacker_wins += 1
                        attacker_success = 1

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
                        "true_label": true_label,
                        "mitre_technique": mitre_label,
                    }
                    writer.writerow(row)

        # 2) Legitimate emails (no attacker)
        for legit_index, legit_text in enumerate(BASE_LEGIT_EMAILS):
            pred, phishing_prob = classify_email(
                vectorizer, clf, legit_text, threshold=threshold
            )

            true_label = 0  # legitimate
            attacker_success = 0  # no attacker in this part

            if pred == 1:
                # Legit email incorrectly flagged as phishing
                fp += 1
            else:
                # Legit email correctly allowed
                tn += 1

            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "round": round_idx,
                "threshold": threshold,
                "base_index": legit_index,
                "base_text": legit_text,
                "rule_name": "legit",
                "variant_index": 0,
                "modified_text": legit_text,
                "predicted_label": pred,
                "phishing_probability": phishing_prob,
                "attacker_success": attacker_success,
                "true_label": true_label,
                "mitre_technique": "N/A - Legit",
            }
            writer.writerow(row)

    # Metrics

    total_samples = tp + fp + tn + fn

    # Detection/bypass rates (phishing only, for attacker vs defender view)
    detection_rate = defender_wins / total_phishing_attacks if total_phishing_attacks > 0 else 0.0
    bypass_rate = attacker_wins / total_phishing_attacks if total_phishing_attacks > 0 else 0.0

    # Standard classification metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0

    summary = {
        "threshold": threshold,
        "total_phishing_attacks": total_phishing_attacks,
        "total_samples": total_samples,
        "attacker_wins": attacker_wins,
        "defender_wins": defender_wins,
        "detection_rate": detection_rate,
        "bypass_rate": bypass_rate,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
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
        "true_label",
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
        "Threshold | Phish Attacks | Defender Wins | Attacker Wins | "
        "Detect Rate | Bypass Rate |  TP |  FP |  TN |  FN | Precision | Recall |   F1  | Accuracy"
    )
    for s in summaries:
        print(
            f"{s['threshold']:8.2f} | "
            f"{s['total_phishing_attacks']:13d} | "
            f"{s['defender_wins']:13d} | "
            f"{s['attacker_wins']:13d} | "
            f"{s['detection_rate']:11.3f} | "
            f"{s['bypass_rate']:11.3f} | "
            f"{s['tp']:3d} | "
            f"{s['fp']:3d} | "
            f"{s['tn']:3d} | "
            f"{s['fn']:3d} | "
            f"{s['precision']:9.3f} | "
            f"{s['recall']:6.3f} | "
            f"{s['f1']:6.3f} | "
            f"{s['accuracy']:8.3f}"
        )

    print(f"\nDetailed log saved to: {SIM_OUTPUT_PATH}")


if __name__ == "__main__":
    # tweak these for experiments
    run_experiments(
        thresholds=[0.5, 0.6, 0.7],
        num_rounds=5,
        num_variants_per_base=2,
    )