# models.py
#
# Small helper module for creating, training, and evaluating
# the baseline classifiers used in this project.
#
# Current models:
# - Logistic Regression + TF-IDF
# - Multinomial Naive Bayes + TF-IDF

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import numpy as np
from utils import get_phishing_class_index


# Label names for pretty printing
LABEL_MAP = {
    0: "legit",
    1: "phishing",
}


def create_logistic_regression():
    """Create the Logistic Regression model used in this project."""
    return LogisticRegression(max_iter=1000, class_weight="balanced")


def create_multinomial_nb():
    """Create the Multinomial Naive Bayes model used in this project."""
    return MultinomialNB()


def train_classifier(classifier, X_train_tfidf, y_train, model_name="Model"):
    """Train a classifier on the TF-IDF training data."""
    print(f"\nTraining {model_name}...")
    classifier.fit(X_train_tfidf, y_train)
    print(f"{model_name} training complete.")
    return classifier


def evaluate_classifier(classifier, X_test_tfidf, y_test, model_name="Model"):
    """Evaluate a classifier on the test split and print the results."""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")
    
    y_pred = classifier.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["legit", "phishing"]))
    
    # Calculate AUC if model supports predict_proba
    auc_score = None
    if hasattr(classifier, "predict_proba"):
        proba = classifier.predict_proba(X_test_tfidf)
        phishing_idx = get_phishing_class_index(classifier)
        y_score = proba[:, phishing_idx]
        auc_score = roc_auc_score(y_test, y_score)
        print(f"ROC AUC Score: {auc_score:.4f}")
    
    return {
        "model_name": model_name,
        "y_pred": y_pred,
        "accuracy": accuracy,
        "auc": auc_score,
    }


def get_model_probabilities(classifier, X_tfidf):
    """Return the class probabilities for a single TF-IDF row."""
    if not hasattr(classifier, "predict_proba"):
        raise ValueError("Classifier does not support predict_proba")
    
    proba = classifier.predict_proba(X_tfidf)[0]
    prob_dict = {}
    
    for label, p in zip(classifier.classes_, proba):
        name = LABEL_MAP.get(label, str(label))
        prob_dict[name] = float(p)
    
    return prob_dict
