# src/attacker_sim.py
#
# Attacker vs Defender simulation with:
# - Legit + phishing emails
# - Single-rule and multi-rule attack chaining
# - Full confusion matrix metrics
# - Threshold testing
# - MITRE mapping
# - XAI explanations logged for attacker wins (false negatives)

import csv
import os
from datetime import datetime, timezone

from mvp_baseline import load_model
from xai_explainer import explain_email


SIM_OUTPUT_DIR = "simulation_results"
SIM_OUTPUT_PATH = os.path.join(SIM_OUTPUT_DIR, "attacker_simulation_log.csv")


# ============== MITRE ATT&CK Mapping Patterns ============== #
# Maps email characteristics to MITRE phishing techniques

MITRE_PATTERNS = {
    'T1566.002': {
        'name': 'Phishing: Link',
        'patterns': ['http://', 'https://', 'www.', 'click here', 'click link', '.com/'],
        'keywords': ['verify', 'confirm', 'validate', 'urgency', 'immediate'],
        'description': 'Email phishing with malicious links'
    },
    'T1566.001': {
        'name': 'Phishing: Attachment',
        'patterns': ['attached', 'document', 'file', 'invoice', 'receipt', 'pdf', 'xlsx', 'docx'],
        'keywords': ['download', 'open', 'review', 'urgent'],
        'description': 'Email phishing with malicious attachments'
    },
    'T1598.003': {
        'name': 'Spearphishing Link (Credible Lookalike)',
        'patterns': ['linkedin', 'facebook', 'google', 'microsoft', 'apple', 'amazon'],
        'keywords': ['social', 'profile', 'connect', 'invite', 'update'],
        'description': 'Spearphishing targeting social media platforms'
    },
    'T1598.001': {
        'name': 'Spearphishing Attachment',
        'patterns': ['invoice', 'receipt', 'contract', 'document', 'spreadsheet'],
        'keywords': ['sign', 'approve', 'review', 'urgent'],
        'description': 'Targeted phishing with official-looking attachments'
    },
    'T1598.002': {
        'name': 'Spearphishing Link (Shortened/Obfuscated)',
        'patterns': ['bit.ly', 'tinyurl', 'goo.gl', 'short', 'tkt.link'],
        'keywords': ['shorten', 'track', 'click'],
        'description': 'Spearphishing with URL shorteners to hide destination'
    },
}


def mitre_mapping(email_text: str, return_all: bool = False):
    """
    Map email to MITRE ATT&CK phishing techniques based on pattern matching.

    Args:
        email_text (str): Email text to analyze.
        return_all (bool): If True, return all matching techniques (multi-label).
                          If False (default), return single highest-confidence match.

    Returns:
        str or list: String technique code and name if return_all=False,
                     List of matching techniques if return_all=True.
                     Always includes primary fallback if no patterns match.

    Example:
        >>> text = "Click here to verify: http://phishing.com"
        >>> mitre_mapping(text)
        'T1566.002 - Phishing: Link'
        >>> mitre_mapping(text, return_all=True)
        ['T1566.002 - Phishing: Link']

    Note:
        - Patterns are matched case-insensitively
        - Multi-label detection shows threat diversity
        - Primary fallback is T1566.001 (generic phishing)
    """
    text_lower = email_text.lower()
    matched_techniques = []

    # Score each technique based on pattern and keyword matches
    technique_scores = {}

    for tech_id, tech_info in MITRE_PATTERNS.items():
        score = 0

        # Check patterns (lower weight)
        for pattern in tech_info['patterns']:
            if pattern in text_lower:
                score += 1

        # Check keywords (higher weight)
        for keyword in tech_info['keywords']:
            if keyword in text_lower:
                score += 2

        if score > 0:
            technique_scores[tech_id] = score
            matched_techniques.append((tech_id, score))

    # Return results
    if return_all:
        # Return all matches sorted by score
        matched_techniques.sort(key=lambda x: x[1], reverse=True)
        if matched_techniques:
            return [f"{tech_id} - {MITRE_PATTERNS[tech_id]['name']}"
                   for tech_id, _ in matched_techniques]
        else:
            return ["T1566.001 - Phishing: Attachment/Generic"]  # Fallback
    else:
        # Return single best match
        if matched_techniques:
            best_match = max(matched_techniques, key=lambda x: x[1])[0]
            return f"{best_match} - {MITRE_PATTERNS[best_match]['name']}"
        else:
            return "T1566.001 - Phishing: Attachment/Generic"  # Fallback


# ---------------- Attacker Rules ---------------- #

def add_urgency(text: str) -> str:
    return (
        "URGENT: Your account may be closed soon. "
        + text
        + " Please respond within 24 hours to avoid losing access."
    )


def spoof_bank(text: str) -> str:
    return (
        "AIB Security Notice: We detected unusual activity on your account. "
        + text
        + " Log in now to confirm your identity."
    )


def spoof_revenue(text: str) -> str:
    return (
        "Irish Revenue: You are eligible for a tax refund. "
        + text
        + " Please follow the link below to claim your refund."
    )


def add_fake_link(text: str) -> str:
    return (
        text
        + " Click here to resolve this issue: http://secure-verification-example.com/login"
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
    """
    Classify a single email using a custom phishing probability threshold.
    """
    X = vectorizer.transform([text])
    proba = clf.predict_proba(X)[0]

    if 1 in clf.classes_:
        phishing_index = list(clf.classes_).index(1)
    else:
        phishing_index = 1

    phishing_prob = float(proba[phishing_index])
    pred_label = 1 if phishing_prob >= threshold else 0

    return pred_label, phishing_prob


# ---------------- Attack Variant Generator ---------------- #

def generate_attack_variants(base_text: str):
    """
    Generate phishing attack variants using:
    - single-rule attacks
    - two-rule chained attacks

    Returns a list of dicts with:
    - attack_type
    - rule_chain
    - modified_text
    """
    variants = []

    # Single-rule attacks
    for rule_name, rule_fn in ATTACK_RULES.items():
        modified_text = rule_fn(base_text)
        variants.append({
            "attack_type": "single_rule",
            "rule_chain": rule_name,
            "modified_text": modified_text,
        })

    # Two-rule chained attacks
    rule_items = list(ATTACK_RULES.items())
    for first_rule_name, first_rule_fn in rule_items:
        for second_rule_name, second_rule_fn in rule_items:
            if first_rule_name == second_rule_name:
                continue

            modified_text = second_rule_fn(first_rule_fn(base_text))
            variants.append({
                "attack_type": "multi_rule",
                "rule_chain": f"{first_rule_name}+{second_rule_name}",
                "modified_text": modified_text,
            })

    return variants


# ---------------- Simulation Core ---------------- #

def run_simulation_for_threshold(
    vectorizer,
    clf,
    threshold: float,
    writer,
    num_rounds: int = 5,
):
    """
    Run the attacker vs defender simulation for one threshold.

    Includes:
    - phishing emails with single and multi-rule attack variants
    - legitimate emails with no attack rules
    - XAI explanations for attacker wins
    """

    tp = fp = tn = fn = 0
    total_phishing_attacks = 0
    attacker_wins = 0
    defender_wins = 0

    for round_idx in range(num_rounds):

        # ---- PHISHING EMAILS ---- #
        for base_index, base_text in enumerate(BASE_PHISHING_EMAILS):
            attack_variants = generate_attack_variants(base_text)

            for variant_index, variant in enumerate(attack_variants):
                total_phishing_attacks += 1

                modified_text = variant["modified_text"]
                attack_type = variant["attack_type"]
                rule_chain = variant["rule_chain"]
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
                    "attack_type": attack_type,
                    "rule_chain": rule_chain,
                    "variant_index": variant_index,
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
                "attack_type": "legit",
                "rule_chain": "none",
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

    detection_rate = (
        defender_wins / total_phishing_attacks if total_phishing_attacks > 0 else 0.0
    )
    bypass_rate = (
        attacker_wins / total_phishing_attacks if total_phishing_attacks > 0 else 0.0
    )

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0

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

def run_experiments(thresholds=None, num_rounds=5):
    """
    Run simulations across multiple thresholds and print summary metrics.
    """
    if thresholds is None:
        thresholds = [0.5, 0.6, 0.7]

    try:
        vectorizer, clf = load_model()
    except FileNotFoundError as e:
        print("\nModel files not found. Please run mvp_baseline.py once first.")
        print("Error:", e)
        return

    os.makedirs(SIM_OUTPUT_DIR, exist_ok=True)

    fieldnames = [
        "timestamp",
        "round",
        "threshold",
        "base_index",
        "base_text",
        "attack_type",
        "rule_chain",
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
                vectorizer=vectorizer,
                clf=clf,
                threshold=thr,
                writer=writer,
                num_rounds=num_rounds,
            )
            summaries.append(summary)

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
    run_experiments(
        thresholds=[0.5, 0.6, 0.7],
        num_rounds=5,
    )