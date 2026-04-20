# compare_models.py
#
# Quick side-by-side model comparison for this project.
# It compares Logistic Regression and Multinomial Naive Bayes
# so I can justify the final model choice with clear metrics.
#
# Run:
#   python src/compare_models.py

import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import joblib

# Ensure src/ is importable
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# Settings

BALANCED_DATA_PATH = "data/processed/english_dataset_balanced.csv"
DATA_PATH = BALANCED_DATA_PATH if os.path.exists(BALANCED_DATA_PATH) else "data/processed/english_dataset.csv"
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

MODEL_DIR = "models"
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
LOGREG_PATH = os.path.join(MODEL_DIR, "logreg_model.joblib")
NB_PATH = os.path.join(MODEL_DIR, "multinomial_nb_model.joblib")

LABEL_MAP = {
    0: "legitimate",
    1: "phishing",
}


# Data loading

def load_data(path, text_col, label_col):
    """Load dataset from CSV."""
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    df = df.dropna(subset=[text_col, label_col])
    
    texts = df[text_col].values
    labels = df[label_col].values
    
    print(f"Loaded {len(texts)} samples")
    return texts, labels


def prepare_data(texts, labels, test_size=0.2, seed=42):
    """Split and vectorize data."""
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    
    # Vectorize with TF-IDF
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=5000,
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"\nData split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {X_train_tfidf.shape[1]}")
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer


# Model training

def train_models(X_train_tfidf, y_train):
    """Train both models."""
    print("\n" + "="*70)
    print("TRAINING MODELS")
    print("="*70)
    
    # Logistic Regression
    print("\n[1/2] Training Logistic Regression...")
    clf_logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf_logreg.fit(X_train_tfidf, y_train)
    print("✓ Logistic Regression trained")
    
    # Naive Bayes
    print("\n[2/2] Training Multinomial Naive Bayes...")
    clf_nb = MultinomialNB()
    clf_nb.fit(X_train_tfidf, y_train)
    print("✓ Multinomial Naive Bayes trained")
    
    return clf_logreg, clf_nb


# Evaluation

def evaluate_model(clf, X_test_tfidf, y_test, model_name):
    """Evaluate a model and return metrics."""
    # Predictions
    y_pred = clf.predict(X_test_tfidf)
    
    # Probabilities (for AUC)
    y_proba = clf.predict_proba(X_test_tfidf)
    if 1 in clf.classes_:
        phishing_idx = list(clf.classes_).index(1)
    else:
        phishing_idx = 1
    y_score = y_proba[:, phishing_idx]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_score)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "y_pred": y_pred,
    }


def print_evaluation_report(results_logreg, results_nb, y_test):
    """Print detailed comparison of both models."""
    print("\n" + "="*70)
    print("MODEL COMPARISON REPORT")
    print("="*70)
    
    # Side-by-side metrics
    print("\n📊 METRICS COMPARISON")
    print("-" * 70)
    print(f"{'Metric':<20} {'Logistic Regression':<25} {'Naive Bayes':<25}")
    print("-" * 70)
    
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    
    for metric in metrics:
        logreg_val = results_logreg[metric]
        nb_val = results_nb[metric]
        
        # Highlight winner
        winner = "🔵 LoGReg" if logreg_val > nb_val else "🟠 NB" if nb_val > logreg_val else "🤝 Tie"
        
        print(f"{metric.upper():<20} {logreg_val:>6.4f}              {nb_val:>6.4f}              {winner}")
    
    # Confusion matrices
    print("\n\n🔍 CONFUSION MATRICES (Phishing Detection)")
    print("-" * 70)
    
    print("\nLogistic Regression:")
    print(f"  True Positives (correctly identified phishing):  {results_logreg['tp']}")
    print(f"  True Negatives (correctly identified legitimate): {results_logreg['tn']}")
    print(f"  False Positives (legitimate marked as phishing): {results_logreg['fp']}")
    print(f"  False Negatives (phishing marked as legitimate): {results_logreg['fn']}")
    
    print("\nMultinomial Naive Bayes:")
    print(f"  True Positives (correctly identified phishing):  {results_nb['tp']}")
    print(f"  True Negatives (correctly identified legitimate): {results_nb['tn']}")
    print(f"  False Positives (legitimate marked as phishing): {results_nb['fp']}")
    print(f"  False Negatives (phishing marked as legitimate): {results_nb['fn']}")
    
    # Classification reports
    print("\n\n📋 DETAILED CLASSIFICATION REPORTS")
    print("-" * 70)
    
    print("\nLogistic Regression:")
    print(classification_report(y_test, results_logreg["y_pred"], 
                              target_names=["Legitimate", "Phishing"]))
    
    print("\nMultinomial Naive Bayes:")
    print(classification_report(y_test, results_nb["y_pred"], 
                              target_names=["Legitimate", "Phishing"]))
    
    # Key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    if results_logreg["accuracy"] > results_nb["accuracy"]:
        print(f"\n✓ Logistic Regression has higher overall accuracy ({results_logreg['accuracy']:.4f})")
    else:
        print(f"\n✓ Naive Bayes has higher overall accuracy ({results_nb['accuracy']:.4f})")
    
    if results_logreg["recall"] > results_nb["recall"]:
        print(f"\n✓ Logistic Regression is better at catching phishing (recall: {results_logreg['recall']:.4f})")
    else:
        print(f"\n✓ Naive Bayes is better at catching phishing (recall: {results_nb['recall']:.4f})")
    
    if results_logreg["precision"] > results_nb["precision"]:
        print(f"\n✓ Logistic Regression has fewer false alarms (precision: {results_logreg['precision']:.4f})")
    else:
        print(f"\n✓ Naive Bayes has fewer false alarms (precision: {results_nb['precision']:.4f})")
    
    print("\n💡 RECOMMENDATION:")
    print("Use the model that best matches your priorities:")
    print("  - Need to catch ALL phishing? → Choose model with higher RECALL")
    print("  - Need fewer false alarms? → Choose model with higher PRECISION")
    print("  - Want balanced performance? → Choose model with higher F1-SCORE")


# ================== MAIN ================== #

def main():
    """Run full model comparison."""
    print("="*70)
    print("PHISHING DETECTION: MODEL COMPARISON")
    print("="*70)
    
    # Load and prepare data
    texts, labels = load_data(DATA_PATH, TEXT_COLUMN, LABEL_COLUMN)
    X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = prepare_data(texts, labels)
    
    # Train both models
    clf_logreg, clf_nb = train_models(X_train_tfidf, y_train)
    
    # Evaluate both models
    print("\n" + "="*70)
    print("EVALUATING MODELS")
    print("="*70)
    
    results_logreg = evaluate_model(clf_logreg, X_test_tfidf, y_test, "Logistic Regression")
    print("✓ Logistic Regression evaluated")
    
    results_nb = evaluate_model(clf_nb, X_test_tfidf, y_test, "Multinomial Naive Bayes")
    print("✓ Multinomial Naive Bayes evaluated")
    
    # Print detailed report
    print_evaluation_report(results_logreg, results_nb, y_test)
    
    # Save to CSV for easy reference
    comparison_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"],
        "Logistic Regression": [
            results_logreg["accuracy"],
            results_logreg["precision"],
            results_logreg["recall"],
            results_logreg["f1"],
            results_logreg["auc"],
        ],
        "Naive Bayes": [
            results_nb["accuracy"],
            results_nb["precision"],
            results_nb["recall"],
            results_nb["f1"],
            results_nb["auc"],
        ],
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    csv_path = os.path.join(PROJECT_ROOT, "model_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    
    print(f"\n✓ Comparison saved to: {csv_path}")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
