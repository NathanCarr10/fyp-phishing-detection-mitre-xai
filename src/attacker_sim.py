# src/attacker_sim.py
#
# Attacker vs Defender simulation with:
# - Legit + phishing emails
# - Full confusion matrix metrics
# - Threshold testing
# - MITRE mapping
# - XAI explanations logged for attacker wins (false negatives)

import csv
import os
from datetime import datetime, timezone

from mvp_baseline import load_model
from xai_explainer import explain_email  # NEW


SIM_OUTPUT_DIR = "simulation_results"
SIM_OUTPUT_PATH = os.path.join(SIM_OUTPUT_DIR, "attacker_simulation_log.csv")


# ---------------- MITRE Mapping ---------------- #

def mitre_mapping(email_text: str) -> str:
    text_lower = email_text.lower()

    if "http://" in text_lower or "https://" in text_lower or "www." in text_lower or "click here" in text_lower:
        return "T1566.002 - Phishing: Link"
    else:
        return "T1566.001 - Phishing: Attachment/Generic"


# ---------------- Attacker Rules ---------------- #

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


# ---------------- Base Emails ---------------- #

BASE_PHISHING_EMAILS = [
    "We have detected suspicious activity on your account.",
    "Your password has expired and must be reset immediately.",
    "There is a problem with your billing information.",
    "We were unable to deliver your package due to an address issue.",
]

BASE_LEGIT_EMAILS = [
    "Hi, just a reminder that our team meeting is at 10am tomorrow.",
    "Please find attached the minutes from last week's project meeting.",
    "Thanks for your help with the report, I really appreciate it.",
    "Here is the updated timetable for the upcoming semester.",
]


# ---------------- Classification Helper ---------------- #

def classify_email(vectorizer, clf, text: str, threshold: float = 0.5):
    X = vectorizer.transform([text])
    proba = clf.predict_proba(X)[0]

    if 1 in clf.classes_:
        phishing_index = list(clf.classes_).index(1)
    else:
        phishing_index = 1

    phishing_prob = float(proba[phishing_index])
    pred_label = 1 if phishing_prob >= threshold else 0

    return pred_label, phishing_prob


# ---------------- Simulation Core ---------------- #

def run_simulation_for_threshold(
    vectorizer,
    clf,
    threshold: float,
    writer,
    num_rounds: int = 5,
    num_variants_per_base: int = 2,
):

    tp = fp = tn = fn = 0
    total_phishing_attacks = 0
    attacker_wins = 0
    defender_wins = 0

    for round_idx in range(num_rounds):

        # ---- PHISHING EMAILS ---- #
        for base_index, base_text in enumerate(BASE_PHISHING_EMAILS):
            for rule_name, rule_fn in ATTACK_RULES.items():
                for variant_idx in range(num_variants_per_base):
                    total_phishing_attacks += 1

                    modified_text = rule_fn(base_text)
                    mitre_label = mitre_mapping(modified_text)

                    pred, phishing_prob = classify_email(
                        vectorizer, clf, modified_text, threshold
                    )

                    true_label = 1
                    attacker_success = 0
                    xai_method = ""
                    xai_top_features = ""

                    if pred == 1:
                        tp += 1
                        defender_wins += 1
                    else:
                        fn += 1
                        attacker_wins += 1
                        attacker_success = 1

                        # ---- CALL XAI FOR ATTACKER WIN ---- #
                        explanation = explain_email(
                            modified_text,
                            num_features=10,
                            threshold=threshold,
                            use_lime=True,
                        )

                        xai_method = explanation["method"]
                        xai_top_features = str(explanation["top_features"])

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
                        "xai_method": xai_method,
                        "xai_top_features": xai_top_features,
                    }
                    writer.writerow(row)

        # ---- LEGIT EMAILS ---- #
        for legit_index, legit_text in enumerate(BASE_LEGIT_EMAILS):

            pred, phishing_prob = classify_email(
                vectorizer, clf, legit_text, threshold
            )

            true_label = 0

            if pred == 1:
                fp += 1
            else:
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
                "attacker_success": 0,
                "true_label": true_label,
                "mitre_technique": "N/A - Legit",
                "xai_method": "",
                "xai_top_features": "",
            }
            writer.writerow(row)

    total_samples = tp + fp + tn + fn

    detection_rate = defender_wins / total_phishing_attacks if total_phishing_attacks else 0
    bypass_rate = attacker_wins / total_phishing_attacks if total_phishing_attacks else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0

    return {
        "threshold": threshold,
        "total_phishing_attacks": total_phishing_attacks,
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


# ---------------- Experiment Runner ---------------- #

def run_experiments(thresholds=None, num_rounds=5, num_variants_per_base=2):

    if thresholds is None:
        thresholds = [0.5, 0.6, 0.7]

    vectorizer, clf = load_model()

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
        "xai_method",
        "xai_top_features",
    ]

    summaries = []

    with open(SIM_OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for thr in thresholds:
            print(f"\nRunning simulation for threshold = {thr:.2f} ...")
            summary = run_simulation_for_threshold(
                vectorizer,
                clf,
                thr,
                writer,
                num_rounds,
                num_variants_per_base,
            )
            summaries.append(summary)

    print("\nSimulation complete.")
    print(f"Detailed log saved to: {SIM_OUTPUT_PATH}")


if __name__ == "__main__":
    run_experiments()