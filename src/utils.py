"""
Shared helper functions for the phishing detection project.
"""

import os
from pathlib import Path
from typing import Optional


def _align_features_for_classifier(X, clf):
    """Align transformed feature matrix with classifier expected feature count."""
    expected_features = getattr(clf, "n_features_in_", None)
    if expected_features is None and hasattr(clf, "coef_"):
        expected_features = clf.coef_.shape[1]

    if expected_features is None:
        return X

    current_features = X.shape[1]
    if current_features == expected_features:
        return X

    if current_features > expected_features:
        return X[:, :expected_features]

    raise ValueError(
        f"Vectorizer produced {current_features} features, but classifier expects {expected_features}."
    )


def get_phishing_class_index(clf) -> int:
    """Return the probability column index for phishing class label 1."""
    if hasattr(clf, "classes_") and 1 in clf.classes_:
        return list(clf.classes_).index(1)
    return 1

def classify_email(vectorizer, clf, text: str, threshold: float = 0.5):
    """Classify one email using the given probability threshold."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Email text must be a non-empty string.")

    X = vectorizer.transform([text])
    X = _align_features_for_classifier(X, clf)
    proba = clf.predict_proba(X)[0]

    phishing_index = get_phishing_class_index(clf)

    phishing_prob = float(proba[phishing_index])
    pred_label = 1 if phishing_prob >= threshold else 0

    return pred_label, phishing_prob


def ensure_directory(dir_path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(dir_path, exist_ok=True)


def get_project_root() -> Path:
    """Return the project root path."""
    return Path(__file__).parent.parent


def load_constants() -> dict:
    """Return the small set of constants shared across the project."""
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
    """Convert a numeric label into a readable name."""
    resolved_label_map = label_map or load_constants().get(
        "LABEL_MAP", {0: "legitimate", 1: "phishing"}
    )
    return resolved_label_map.get(label, str(label))


__all__ = [
    'classify_email',
    'ensure_directory',
    'get_phishing_class_index',
    'get_project_root',
    'load_constants',
    'get_label_name',
]
