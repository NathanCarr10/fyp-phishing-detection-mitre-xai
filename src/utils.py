"""
Shared utility functions for phishing detection system.

This module contains common functions used across multiple modules to avoid code duplication.
"""

import os
from pathlib import Path
from typing import Optional

def classify_email(vectorizer, clf, text: str, threshold: float = 0.5):
    """
    Classify an email as phishing or legitimate with custom threshold.

    Args:
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer for feature extraction.
        clf (LogisticRegression): Trained classifier model.
        text (str): Email text to classify.
        threshold (float): Custom probability threshold for positive class (default: 0.5).
                          Predictions >= threshold are classified as phishing (1).

    Returns:
        tuple: (pred_label, phishing_prob) where:
               - pred_label: Integer (0=legitimate, 1=phishing) based on threshold
               - phishing_prob: Float [0, 1] probability of phishing class

    Raises:
        ValueError: If email text is empty or None.
        AttributeError: If classifier does not support predict_proba.

    Example:
        >>> vectorizer, clf = load_model()
        >>> label, prob = classify_email(vectorizer, clf, email_text, threshold=0.6)
        >>> print(f"Email is {'phishing' if label else 'legitimate'} (prob={prob:.3f})")

    Note:
        - Threshold can be tuned to balance false positives vs false negatives
        - Default 0.5 gives equal weight to both classes
        - Higher threshold → more conservative (fewer false positives)
        - Lower threshold → more sensitive (fewer false negatives)
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Email text must be a non-empty string.")

    X = vectorizer.transform([text])
    proba = clf.predict_proba(X)[0]

    # Find the index of the phishing class (label=1)
    if 1 in clf.classes_:
        phishing_index = list(clf.classes_).index(1)
    else:
        phishing_index = 1

    phishing_prob = float(proba[phishing_index])
    pred_label = 1 if phishing_prob >= threshold else 0

    return pred_label, phishing_prob


def ensure_directory(dir_path: str) -> None:
    """
    Create directory if it doesn't exist (mkdir -p equivalent).

    Args:
        dir_path (str): Path to directory to create.

    Returns:
        None

    Side Effects:
        Creates directory and any parent directories as needed.

    Example:
        >>> ensure_directory("output/results/figures")
    """
    os.makedirs(dir_path, exist_ok=True)


def get_project_root() -> Path:
    """
    Get the project root directory path.

    Returns:
        Path: Path object pointing to project root (parent of src/).

    Example:
        >>> root = get_project_root()
        >>> data_path = root / "data" / "processed" / "dataset.csv"
    """
    return Path(__file__).parent.parent


def load_constants() -> dict:
    """
    Load commonly used constants.

    Returns:
        dict: Dictionary containing:
              - TFIDF_MAX_FEATURES: Maximum features for vectorizer
              - TFIDF_STOP_WORDS: Stop words for vectorizer
              - LR_MAX_ITER: Max iterations for Logistic Regression
              - LABEL_MAP: Mapping of integer labels to names
              - DEFAULT_THRESHOLD: Default classification threshold

    Example:
        >>> consts = load_constants()
        >>> max_features = consts['TFIDF_MAX_FEATURES']
    """
    return {
        'TFIDF_MAX_FEATURES': 5000,
        'TFIDF_STOP_WORDS': 'english',
        'LR_MAX_ITER': 1000,
        'LABEL_MAP': {
            0: 'legitimate',
            1: 'phishing',
        },
        'DEFAULT_THRESHOLD': 0.5,
        'RANDOM_SEED': 42,
    }


def get_label_name(label: int, label_map: Optional[dict] = None) -> str:
    """
    Convert numeric label to human-readable name.

    Args:
        label (int): Numeric label (0 or 1).
        label_map (dict, optional): Custom mapping. Defaults to standard map.

    Returns:
        str: Human-readable label name.

    Example:
        >>> get_label_name(1)
        'phishing'
    """
    if label_map is None:
        label_map = load_constants()['LABEL_MAP']
    
    assert label_map is not None, "label_map should not be None"
    return label_map.get(label, str(label))


__all__ = [
    'classify_email',
    'ensure_directory',
    'get_project_root',
    'load_constants',
    'get_label_name',
]
