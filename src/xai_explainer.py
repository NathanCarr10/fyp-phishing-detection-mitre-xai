# src/xai_explainer.py
#
# Explains why the model predicted phishing or legitimate.
#
# It loads the saved model, tries LIME first, and falls back to a simple
# linear explanation when LIME is not available.

from __future__ import annotations

import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from mvp_baseline import DATA_PATH
from mvp_baseline import TEXT_COLUMN
from mvp_baseline import load_model  # model loader used by the project
from utils import _align_features_for_classifier
from utils import get_phishing_class_index


# Optional LIME support
try:
    from lime.lime_text import LimeTextExplainer  # type: ignore
    _LIME_AVAILABLE = True
except ImportError:
    _LIME_AVAILABLE = False


# Optional SHAP support
try:
    import shap  # type: ignore
    _SHAP_AVAILABLE = True
except ImportError:
    shap = None  # type: ignore[assignment]
    _SHAP_AVAILABLE = False


# Internal model cache


_vectorizer = None
_clf = None
_shap_linear_explainer = None


def _get_model():
    """Load the model once and reuse it on later calls."""
    global _vectorizer, _clf
    if _vectorizer is None or _clf is None:
        _vectorizer, _clf = load_model()
    return _vectorizer, _clf


def _get_phishing_class_index(clf) -> int:
    """Return the probability column for the phishing class."""
    return get_phishing_class_index(clf)


def _get_shap_linear_explainer():
    """Build and cache a SHAP linear explainer using a small text background."""
    global _shap_linear_explainer
    if _shap_linear_explainer is not None:
        return _shap_linear_explainer

    if not _SHAP_AVAILABLE:
        raise RuntimeError("SHAP is not available in this environment.")
    if shap is None:
        raise RuntimeError("SHAP module failed to load.")

    vectorizer, clf = _get_model()

    # Keep background small so app startup stays responsive.
    dataset = pd.read_csv(DATA_PATH)
    text_series = dataset[TEXT_COLUMN].dropna().astype(str)
    if text_series.empty:
        raise RuntimeError("No background text is available for SHAP.")

    background_texts = text_series.head(200).tolist()
    X_background = vectorizer.transform(background_texts)
    X_background = _align_features_for_classifier(X_background, clf)

    _shap_linear_explainer = shap.LinearExplainer(clf, X_background)
    return _shap_linear_explainer


# LIME explanation


def _explain_with_lime(
    text: str,
    num_features: int = 10,
) -> List[Tuple[str, float]]:
    """Use LIME to show which words pushed the prediction."""
    if not _LIME_AVAILABLE:
        raise RuntimeError("LIME is not available in this environment.")

    vectorizer, clf = _get_model()
    phishing_index = _get_phishing_class_index(clf)

    # LIME expects a function that returns class probabilities for a list of texts.
    def predict_proba(texts: List[str]):
        X = vectorizer.transform(texts)
        X = _align_features_for_classifier(X, clf)
        return clf.predict_proba(X)

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

    feature_weights = explanation.as_list(label=phishing_index)
    return feature_weights


# Linear fallback


def _explain_with_linear_weights(
    text: str,
    num_features: int = 10,
) -> List[Tuple[str, float]]:
    """Use the model coefficients as a simple fallback explanation."""
    vectorizer, clf = _get_model()
    phishing_index = _get_phishing_class_index(clf)

    X = vectorizer.transform([text])
    X = _align_features_for_classifier(X, clf)
    X_dense = X.toarray()[0]

    if not hasattr(clf, "coef_"):
        return []

    if clf.coef_.shape[0] == 1:
        coef = clf.coef_[0]
    else:
        coef = clf.coef_[phishing_index]

    try:
        feature_names = vectorizer.get_feature_names_out()
    except AttributeError:
        feature_names = vectorizer.get_feature_names()

    contributions: List[Tuple[str, float]] = []
    for idx, value in enumerate(X_dense):
        if value == 0.0:
            continue
        contrib = float(value * coef[idx])
        contributions.append((feature_names[idx], contrib))

    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    return contributions[:num_features]


def _explain_with_shap(
    text: str,
    num_features: int = 10,
) -> List[Tuple[str, float]]:
    """Use SHAP values to show the strongest feature contributions."""
    if not _SHAP_AVAILABLE:
        raise RuntimeError("SHAP is not available in this environment.")

    vectorizer, clf = _get_model()
    phishing_index = _get_phishing_class_index(clf)

    X = vectorizer.transform([text])
    X = _align_features_for_classifier(X, clf)
    X_dense = X.toarray()[0]

    explainer = _get_shap_linear_explainer()
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_vector = np.ravel(shap_values[phishing_index])
    else:
        shap_array = np.asarray(shap_values)
        if shap_array.ndim == 3:
            shap_vector = np.ravel(shap_array[0, :, phishing_index])
        else:
            shap_vector = np.ravel(shap_array[0])

    try:
        feature_names = vectorizer.get_feature_names_out()
    except AttributeError:
        feature_names = vectorizer.get_feature_names()

    contributions: List[Tuple[str, float]] = []
    for idx, value in enumerate(X_dense):
        if value == 0.0:
            continue
        contributions.append((feature_names[idx], float(shap_vector[idx])))

    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return contributions[:num_features]


# Public API


def explain_email(
    text: str,
    num_features: int = 10,
    threshold: float = 0.5,
    use_lime: bool = True,
    use_shap: bool = False,
) -> Dict[str, Any]:
    """Predict a single email and return the explanation in a dict."""
    vectorizer, clf = _get_model()
    phishing_index = _get_phishing_class_index(clf)

    X = vectorizer.transform([text])
    X = _align_features_for_classifier(X, clf)
    proba = clf.predict_proba(X)[0]
    phishing_prob = float(proba[phishing_index])

    pred_label = 1 if phishing_prob >= threshold else 0

    features: List[Tuple[str, float]] = []
    method = "none"

    if use_shap and _SHAP_AVAILABLE:
        try:
            features = _explain_with_shap(text, num_features=num_features)
            method = "shap"
        except Exception:
            features = _explain_with_linear_weights(text, num_features=num_features)
            method = "linear"
    elif use_lime and _LIME_AVAILABLE:
        try:
            features = _explain_with_lime(text, num_features=num_features)
            method = "lime"
        except Exception:
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


# Command-line test helper


def _cli():
    """Simple command-line test loop for manual checks."""
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