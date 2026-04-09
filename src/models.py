#
# models.py
#
# Model definitions and utilities for phishing detection.
# Supports multiple classifier algorithms for comparison.
#
# Current models:
#  - Logistic Regression with TF-IDF
#  - Multinomial Naive Bayes with TF-IDF

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import numpy as np


# Label names for pretty printing
LABEL_MAP = {
    0: "legit",
    1: "phishing",
}


def create_logistic_regression():
    """Create and return a Logistic Regression classifier."""
    return LogisticRegression(max_iter=1000, class_weight="balanced")


def create_multinomial_nb():
    """Create and return a Multinomial Naive Bayes classifier."""
    return MultinomialNB()


def train_classifier(classifier, X_train_tfidf, y_train, model_name="Model"):
    """
    Train a classifier on TF-IDF transformed text data.
    
    Args:
        classifier: Scikit-learn classifier object
        X_train_tfidf: Sparse matrix of TF-IDF features (training)
        y_train: Array of training labels
        model_name: Name of model for logging (e.g., "Logistic Regression")
    
    Returns:
        Trained classifier object
    """
    print(f"\nTraining {model_name}...")
    classifier.fit(X_train_tfidf, y_train)
    print(f"{model_name} training complete.")
    return classifier


def evaluate_classifier(classifier, X_test_tfidf, y_test, model_name="Model"):
    """
    Evaluate classifier on test data and print metrics.
    
    Args:
        classifier: Trained classifier
        X_test_tfidf: Sparse matrix of TF-IDF features (test)
        y_test: Array of test labels
        model_name: Name of model for logging
    
    Returns:
        dict: Contains predictions, accuracy, and AUC score
    """
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
        if 1 in classifier.classes_:
            phishing_idx = list(classifier.classes_).index(1)
        else:
            phishing_idx = 1
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
    """
    Get probability predictions from classifier.
    
    Args:
        classifier: Trained classifier with predict_proba method
        X_tfidf: Sparse matrix of TF-IDF features
    
    Returns:
        dict: Probabilities for each class
    """
    if not hasattr(classifier, "predict_proba"):
        raise ValueError("Classifier does not support predict_proba")
    
    proba = classifier.predict_proba(X_tfidf)[0]
    prob_dict = {}
    
    for label, p in zip(classifier.classes_, proba):
        name = LABEL_MAP.get(label, str(label))
        prob_dict[name] = float(p)
    
    return prob_dict
