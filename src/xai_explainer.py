# src/xai_explainer.py
#
# XAI explainer module for the phishing detection model.
#
# - Loads the saved TF-IDF vectorizer and classifier from mvp_baseline.py
# - Provides a high-level explain_email(...) function
# - Tries to use LIME for text explanations if available
# - Falls back to a simple linear-weight explanation if LIME is not installed
#
# Usage (CLI):
#   python src/xai_explainer.py
#   -> paste an email and see prediction + top features
#
# Usage (in code):
#   from xai_explainer import explain_email
#   result = explain_email("Some email text here")
#   print(result["pred_label"], result["phishing_probability"])
#   for feat in result["top_features"]:
#       print(feat["term"], feat["weight"])

from __future__ import annotations

import sys
from typing import Any, Dict, List, Tuple

from mvp_baseline import load_model  # your existing model loader


# Try to import LIME if available
try:
    from lime.lime_text import LimeTextExplainer  # type: ignore
    _LIME_AVAILABLE = True
except ImportError:
    _LIME_AVAILABLE = False


# --- Internal: model loading / caching -------------------------------------------------


_vectorizer = None
_clf = None


def _get_model():
    """
    Lazy-load and cache the TF-IDF vectorizer and classifier.

    Returns
    -------
    vectorizer, clf
    """
    global _vectorizer, _clf
    if _vectorizer is None or _clf is None:
        _vectorizer, _clf = load_model()
    return _vectorizer, _clf


def _get_phishing_class_index(clf) -> int:
    """
    Get the index of the phishing class (assumed to be label 1).

    If clf.classes_ does not contain 1, fallback to index 1.
    """
    if hasattr(clf, "classes_") and 1 in clf.classes_:
        return list(clf.classes_).index(1)
    # Fallback: assume second column is the "positive" phishing class
    return 1


# --- LIME-based explanation ------------------------------------------------------------


def _explain_with_lime(
    text: str,
    num_features: int = 10,
) -> List[Tuple[str, float]]:
    """
    Explain a single email using LIME (if available).

    Parameters
    ----------
    text : str
        The email text to explain.
    num_features : int
        Number of top features (words) to return.

    Returns
    -------
    List[Tuple[str, float]]
        List of (term, weight) pairs from LIME, sorted by importance.
    """
    if not _LIME_AVAILABLE:
        raise RuntimeError("LIME is not available in this environment.")

    vectorizer, clf = _get_model()
    phishing_index = _get_phishing_class_index(clf)

    # LIME expects a callable that takes a list of texts and returns
    # an array of probabilities for each class.
    def predict_proba(texts: List[str]):
        X = vectorizer.transform(texts)
        return clf.predict_proba(X)

    # Class names: align indices with clf.classes_
    if hasattr(clf, "classes_"):
        class_labels = clf.classes_
        class_names = [str(c) for c in class_labels]
    else:
        class_names = ["0", "1"]

    explainer = LimeTextExplainer(class_names=class_names)  # type: ignore
    explanation = explainer.explain_instance(
        text,
        predict_proba,
        num_features=num_features,
        labels=[phishing_index],
    )

    # LIME's as_list(label=...) returns list[(feature, weight)]
    feature_weights = explanation.as_list(label=phishing_index)
    return feature_weights


# --- Simple linear-weight explanation (fallback) ---------------------------------------


def _explain_with_linear_weights(
    text: str,
    num_features: int = 10,
) -> List[Tuple[str, float]]:
    """
    Simple explanation based on linear model weights and TF-IDF features.

    This does not require LIME and works if:
      - vectorizer is a scikit-learn text vectorizer with get_feature_names_out()
      - clf is a linear model with coef_ attribute (e.g. LogisticRegression)

    Parameters
    ----------
    text : str
        The email text to explain.
    num_features : int
        Number of top features (words) to return.

    Returns
    -------
    List[Tuple[str, float]]
        List of (term, contribution) pairs sorted by absolute contribution
        to the phishing score.
    """
    vectorizer, clf = _get_model()
    phishing_index = _get_phishing_class_index(clf)

    # Transform text to TF-IDF vector
    X = vectorizer.transform([text])
    # Convert to dense to inspect per-feature contributions
    X_dense = X.toarray()[0]

    # Get coefficients for the phishing class
    if not hasattr(clf, "coef_"):
        # If no coef_ attribute, we can't do this explanation
        return []

    coef = clf.coef_[phishing_index]

    # Feature names
    try:
        feature_names = vectorizer.get_feature_names_out()
    except AttributeError:
        feature_names = vectorizer.get_feature_names()

    # Contribution of each feature = tf-idf value * coefficient
    contributions: List[Tuple[str, float]] = []
    for idx, value in enumerate(X_dense):
        if value == 0.0:
            continue  # feature not present in this email
        contrib = float(value * coef[idx])
        contributions.append((feature_names[idx], contrib))

    # Sort by absolute contribution, descending
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    # Return top-k
    return contributions[:num_features]


# --- High-level public API -------------------------------------------------------------


def explain_email(
    text: str,
    num_features: int = 10,
    threshold: float = 0.5,
    use_lime: bool = True,
) -> Dict[str, Any]:
    """
    Generate an explanation for a single email.

    This function:
      - Loads the vectorizer and classifier
      - Predicts phishing probability and label using the given threshold
      - Computes top contributing features either with LIME (if available)
        or with a simple linear-weight fallback.
      - Returns a structured dict suitable for logging or JSON.

    Parameters
    ----------
    text : str
        The email text to classify and explain.
    num_features : int
        Number of top features (words) to include in the explanation.
    threshold : float
        Decision threshold on P(phishing) for classifying as phishing (1).
    use_lime : bool
        If True, attempt to use LIME if installed. If LIME is not available
        or use_lime is False, fallback to linear-weight explanation.

    Returns
    -------
    Dict[str, Any]
        {
            "text": str,
            "pred_label": int,
            "phishing_probability": float,
            "threshold": float,
            "is_phishing": bool,
            "top_features": List[{"term": str, "weight": float}],
            "method": "lime" or "linear",
        }
    """
    vectorizer, clf = _get_model()
    phishing_index = _get_phishing_class_index(clf)

    # Predict probability
    X = vectorizer.transform([text])
    proba = clf.predict_proba(X)[0]
    phishing_prob = float(proba[phishing_index])

    # Apply threshold
    pred_label = 1 if phishing_prob >= threshold else 0

    # Get feature importances
    features: List[Tuple[str, float]] = []
    method = "none"

    if use_lime and _LIME_AVAILABLE:
        try:
            features = _explain_with_lime(text, num_features=num_features)
            method = "lime"
        except Exception:
            # If anything goes wrong with LIME, fallback gracefully
            features = _explain_with_linear_weights(text, num_features=num_features)
            method = "linear"
    else:
        features = _explain_with_linear_weights(text, num_features=num_features)
        method = "linear"

    top_features_struct = [
        {"term": term, "weight": float(weight)} for term, weight in features
    ]

    return {
        "text": text,
        "pred_label": int(pred_label),
        "phishing_probability": phishing_prob,
        "threshold": float(threshold),
        "is_phishing": bool(pred_label == 1),
        "top_features": top_features_struct,
        "method": method,
    }


# --- Simple CLI for manual testing -----------------------------------------------------


def _cli():
    """
    Simple command-line interface for quick testing.

    Run:
        python src/xai_explainer.py

    Then paste an email and press Enter (Ctrl+D / Ctrl+Z to exit).
    """
    print("XAI Email Explainer")
    print("-------------------")
    print("Paste an email below. Press Ctrl+C to exit.\n")

    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue

        result = explain_email(text, num_features=10, threshold=0.5, use_lime=True)

        print("\nPrediction:")
        print(f"  Label: {'phishing' if result['is_phishing'] else 'legit'}")
        print(f"  P(phishing): {result['phishing_probability']:.3f}")
        print(f"  Method: {result['method']}")
        print("\nTop features:")
        for feat in result["top_features"]:
            print(f"  {feat['term']:<20} {feat['weight']:+.4f}")
        print("\nPaste another email or Ctrl+C to quit.\n")


if __name__ == "__main__":
    _cli()