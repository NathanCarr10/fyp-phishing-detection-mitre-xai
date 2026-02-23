# src/attacker_sim.py
#
# Simple attacker vs defender simulation.
#
# - Uses the saved model from mvp_baseline.py
# - Starts from simple phishing-style base messages
# - Applies simple "attack rules" to modify them
# - Sends each variant through the model
# - Logs whether the attacker "won" (bypassed detection) or not
# - Adds a simple MITRE ATT&CK tag (T1566.*) based on the email content

import csv
import os
from datetime import datetime

from mvp_baseline import predict_single_email  # uses saved model

# Output CSV for logging results
SIM_OUTPUT_DIR = "simulation_results"
SIM_OUTPUT_PATH = os.path.join(SIM_OUTPUT_DIR, "attacker_simulation_log.csv")


# MITRE mapping helper

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


# Simulation

def run_simulation(num_variants_per_base: int = 3):
    """
    Run a simple attacker vs defender simulation.
    For each base phishing email:
      - apply each attack rule
      - call the model
      - log whether the attacker "won" (phishing predicted as legit)
    """
    os.makedirs(SIM_OUTPUT_DIR, exist_ok=True)

    # Prepare CSV logging
    fieldnames = [
        "timestamp",
        "base_index",
        "base_text",
        "rule_name",
        "modified_text",
        "predicted_label",
        "predicted_label_name",
        "phishing_probability",
        "attacker_success",
        "mitre_technique",
    ]

    with open(SIM_OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        total_attacks = 0
        attacker_wins = 0
        defender_wins = 0

        for base_index, base_text in enumerate(BASE_PHISHING_EMAILS):
            for rule_name, rule_fn in ATTACK_RULES.items():
                for _ in range(num_variants_per_base):
                    total_attacks += 1

                    modified_text = rule_fn(base_text)
                    mitre_label = mitre_mapping(modified_text)

                    try:
                        pred, pred_name, probs = predict_single_email(modified_text)
                    except FileNotFoundError as e:
                        print("\nModel files not found. Please run mvp_baseline.py once to train and save the model.")
                        print("Error:", e)
                        return

                    phishing_prob = probs.get("phishing", 0.0)

                    # Attacker "wins" if the model says this is legit (0)
                    attacker_success = int(pred == 0)
                    if attacker_success:
                        attacker_wins += 1
                    else:
                        defender_wins += 1

                    row = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "base_index": base_index,
                        "base_text": base_text,
                        "rule_name": rule_name,
                        "modified_text": modified_text,
                        "predicted_label": pred,
                        "predicted_label_name": pred_name,
                        "phishing_probability": phishing_prob,
                        "attacker_success": attacker_success,
                        "mitre_technique": mitre_label,
                    }
                    writer.writerow(row)

        print("\nSimulation finished.")
        print("Total simulated attacks:   ", total_attacks)
        print("Attacker wins (bypassed):  ", attacker_wins)
        print("Defender wins (detected):  ", defender_wins)
        if total_attacks > 0:
            print("Detection rate:            ", f"{defender_wins / total_attacks:.3f}")
            print("Bypass rate (attacker):    ", f"{attacker_wins / total_attacks:.3f}")
        print(f"\nDetailed log saved to: {SIM_OUTPUT_PATH}")


if __name__ == "__main__":
    run_simulation(num_variants_per_base=2)
