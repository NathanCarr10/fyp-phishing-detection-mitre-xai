import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import joblib

try:
    from lime.lime_text import LimeTextExplainer
    _LIME_AVAILABLE = True
except ImportError:
    _LIME_AVAILABLE = False

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

# Paths and column names
BALANCED_DATA_PATH = "data/processed/english_dataset_balanced.csv"
DATA_PATH = BALANCED_DATA_PATH if os.path.exists(BALANCED_DATA_PATH) else "data/processed/english_dataset.csv"
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

# Paths for saving model and vectorizer
MODEL_DIR = "models"
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, "logreg_model.joblib")
NB_MODEL_PATH = os.path.join(MODEL_DIR, "multinomial_nb_model.joblib")

# label names for printing and explanations
LABEL_MAP = {
    0: "legit",
    1: "phishing",
}


def load_data(path, text_col, label_col):
    """
    Load email dataset from CSV file.

    Args:
        path (str): Path to the CSV file containing email data.
        text_col (str): Name of the column containing email text.
        label_col (str): Name of the column containing binary labels (0/1).

    Returns:
        tuple: (texts, labels) where texts is array of email strings,
               labels is array of binary labels.

    Raises:
        ValueError: If specified columns not found in CSV file.
        FileNotFoundError: If CSV file does not exist.
    """
    print("Loading dataset from:", path)
    df = pd.read_csv(path)

    print("\nFirst few rows of the dataset:")
    print(df.head())

    print("\nColumns in the dataset:")
    print(df.columns)

    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"\nColumn names not found.\n"
            f"Current TEXT_COLUMN='{text_col}', LABEL_COLUMN='{label_col}'\n"
            f"Available columns: {list(df.columns)}\n"
            f"Edit TEXT_COLUMN and LABEL_COLUMN at the top of the file."
        )

    df = df.dropna(subset=[text_col, label_col])

    texts = df[text_col].values
    labels = df[label_col].values

    print(f"\nNumber of samples after dropping missing values: {len(texts)}")
    return texts, labels


def split_data(texts, labels, test_size=0.2, seed=42):
    """Split texts and labels into train and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    print("\nTraining samples:", len(X_train))
    print("Test samples:", len(X_test))
    return X_train, X_test, y_train, y_test


def vectorize_text(X_train, X_test):
    """Fit TF-IDF on training data and transform both train and test."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=5000,
    )

    print("\nFitting TF-IDF vectorizer on training data...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("TF-IDF train matrix shape:", X_train_tfidf.shape)
    print("TF-IDF test matrix shape:", X_test_tfidf.shape)

    return vectorizer, X_train_tfidf, X_test_tfidf


def train_model(clf, X_train_tfidf, y_train):
    """
    Train a classifier on TF-IDF transformed data.
    
    Args:
        clf: Scikit-learn classifier object (pre-instantiated)
        X_train_tfidf: Sparse matrix of TF-IDF features
        y_train: Array of training labels
    
    Returns:
        Trained classifier object
    """
    print(f"\nTraining {clf.__class__.__name__}...")
    clf.fit(X_train_tfidf, y_train)
    return clf


def evaluate_model(clf, X_test_tfidf, y_test):
    """Evaluate model on test data and print metrics."""
    print("\nEvaluating on test data...")
    y_pred = clf.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return y_pred


def show_example_predictions(clf, X_test, X_test_tfidf, y_test, y_pred, num_examples=5):
    """Print a few test emails with true vs predicted labels."""
    print("\nSome example predictions:\n")

    n = min(num_examples, len(X_test))
    for i in range(n):
        print("Email text:")
        print(X_test[i][:200].replace("\n", " "))

        true_label = y_test[i]
        pred_label = y_pred[i]
        true_name = LABEL_MAP.get(true_label, str(true_label))
        pred_name = LABEL_MAP.get(pred_label, str(pred_label))

        print("True label:   ", true_name, f"({true_label})")
        print("Predicted:    ", pred_name, f"({pred_label})")

        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_test_tfidf[i])[0]
            print("Class probabilities:")
            for label, p in zip(clf.classes_, proba):
                label_name = LABEL_MAP.get(label, str(label))
                print(f"  {label_name} ({label}): {p:.3f}")

        print("-" * 60)


def explain_with_lime(clf, vectorizer, text_sample):
    """Use LIME to explain one email."""
    if not _LIME_AVAILABLE:
        print("\nLIME not installed. Skipping LIME explanation.")
        return

    def predict_proba(text_list):
        X = vectorizer.transform(text_list)
        return clf.predict_proba(X)

    class_names = [LABEL_MAP.get(c, str(c)) for c in clf.classes_]
    explainer = LimeTextExplainer(class_names=class_names)

    exp = explainer.explain_instance(
        text_sample,
        predict_proba,
        num_features=10,
    )

    print("\nLIME explanation (word, weight):")
    for word, weight in exp.as_list():
        print(f"  {word}: {weight:.4f}")

    try:
        html = exp.as_html()
        with open("lime_explanation.html", "w", encoding="utf-8") as f:
            f.write(html)
        print("\nSaved detailed HTML explanation to lime_explanation.html")
    except Exception as e:
        print("\nCould not save HTML explanation:", e)


# SHAP-related functions

def sample_rows(X, n=2000, seed=42):
    """Return a random sample of rows from a sparse matrix."""
    n = min(n, X.shape[0])
    rng = np.random.RandomState(seed)
    idx = rng.choice(X.shape[0], size=n, replace=False)
    return X[idx], idx


def build_shap_explainer(clf, X_train_tfidf, background_size=2000):
    """Create a SHAP explainer using a small background sample."""
    if not _SHAP_AVAILABLE:
        raise RuntimeError("SHAP not installed. Cannot build SHAP explainer.")

    print("\nBuilding SHAP explainer (using a small background sample)...")
    X_bg, _ = sample_rows(X_train_tfidf, n=background_size, seed=42)
    explainer = shap.LinearExplainer(clf, X_bg)
    return explainer


def _select_shap_array(shap_values, clf, target_label=None):
    """Handle SHAP output for binary/multiclass."""
    if isinstance(shap_values, list):
        if target_label is not None and target_label in clf.classes_:
            idx = list(clf.classes_).index(target_label)
        else:
            idx = 0
        shap_array = shap_values[idx]
    else:
        shap_array = shap_values

    return np.array(shap_array)


def explain_with_shap_global(explainer, clf, X_train_tfidf, feature_names, top_n=20, sample_size=2000):
    """Print top features globally using SHAP (sample-based to save memory)."""
    print("\nComputing SHAP global feature importance...")

    X_sample, _ = sample_rows(X_train_tfidf, n=sample_size, seed=42)
    print(f"Using a sample of {X_sample.shape[0]} emails for global SHAP.")

    shap_values = explainer.shap_values(X_sample)
    target_label = 1 if 1 in clf.classes_ else None
    shap_array = _select_shap_array(shap_values, clf, target_label=target_label)

    mean_abs_shap = np.mean(np.abs(shap_array), axis=0)
    top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]

    print(f"\nTop {top_n} global features by SHAP (sample-based):")
    for idx in top_indices:
        print(f"  {feature_names[idx]}: {mean_abs_shap[idx]:.6f}")


def explain_with_shap_local(explainer, clf, X_test_tfidf, feature_names, index, y_test=None, top_n=10):
    """Explain one test email with SHAP (local explanation)."""
    print(f"\nComputing SHAP local explanation for test index {index}...")

    shap_values = explainer.shap_values(X_test_tfidf[index])
    target_label = 1 if 1 in clf.classes_ else None
    shap_array = _select_shap_array(shap_values, clf, target_label=target_label)

    shap_flat = np.ravel(shap_array)
    top_indices = np.argsort(np.abs(shap_flat))[-top_n:][::-1]

    if y_test is not None:
        true_label = y_test[index]
        print("True label for this email:", LABEL_MAP.get(true_label, str(true_label)))

    print(f"\nTop {top_n} local SHAP features for this email:")
    for idx in top_indices:
        print(f"  {feature_names[idx]}: {shap_flat[idx]:.6f}")


# ROC/AUC function

def plot_roc_auc(clf, X_test_tfidf, y_test, out_path="roc_curve.png"):
    """Compute AUC and save ROC curve to a PNG file."""
    # We need probabilities for the phishing class (label 1)
    if not hasattr(clf, "predict_proba"):
        print("\nModel does not support predict_proba, cannot compute ROC/AUC.")
        return None

    proba = clf.predict_proba(X_test_tfidf)

    # Find which column is class "1"
    if 1 in clf.classes_:
        phishing_index = list(clf.classes_).index(1)
    else:
        # fallback: assume second column is "positive"
        phishing_index = 1

    y_score = proba[:, phishing_index]

    auc = roc_auc_score(y_test, y_score)
    fpr, tpr, thresholds = roc_curve(y_test, y_score)

    print(f"\nROC AUC score (phishing=1): {auc:.4f}")

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")  # baseline
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"Saved ROC curve plot to: {out_path}")
    return auc


# Model save/load + single email prediction

def save_model(vectorizer, clf, model_path=MODEL_PATH):
    """
    Save the trained vectorizer and model to disk.
    
    Args:
        vectorizer: Fitted TfidfVectorizer
        clf: Trained classifier
        model_path: Path to save the model (default: Logistic Regression path)
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(clf, model_path)
    print(f"\nSaved vectorizer to: {VECTORIZER_PATH}")
    print(f"Saved model to:      {model_path}")


def load_model(model_path=MODEL_PATH):
    """
    Load the vectorizer and model from disk.
    
    Args:
        model_path: Path to load the model from (default: Logistic Regression path)
    
    Returns:
        tuple: (vectorizer, classifier)
    """
    if not (os.path.exists(VECTORIZER_PATH) and os.path.exists(model_path)):
        raise FileNotFoundError(
            f"Model files not found. Train the model and run save_model() first.\n"
            f"Looking for: {VECTORIZER_PATH} and {model_path}"
        )
    vectorizer = joblib.load(VECTORIZER_PATH)
    clf = joblib.load(model_path)
    return vectorizer, clf


def predict_single_email(text: str, model_path=MODEL_PATH):
    """
    Classify a single email as phishing or legitimate.

    Args:
        text (str): Email text to classify.
        model_path (str): Path to the model file. 
                         Default: Logistic Regression
                         Alternative: NB_MODEL_PATH for Naive Bayes

    Returns:
        tuple: (pred_label, pred_name, prob_dict) where:
               - pred_label: Integer label (0=legitimate, 1=phishing)
               - pred_name: String label name
               - prob_dict: Dict with probabilities {"legit": ..., "phishing": ...}

    Raises:
        FileNotFoundError: If model files not found. Run mvp_baseline.py first.

    Note:
        High-level convenience function for single email prediction.
    """
    vectorizer, clf = load_model(model_path=model_path)
    X = vectorizer.transform([text])
    proba = clf.predict_proba(X)[0]
    pred = clf.predict(X)[0]

    # Map probabilities to label names
    prob_dict = {}
    for label, p in zip(clf.classes_, proba):
        name = LABEL_MAP.get(label, str(label))
        prob_dict[name] = float(p)

    pred_name = LABEL_MAP.get(pred, str(pred))

    return int(pred), pred_name, prob_dict


def main():
    # 1. Load and split data
    texts, labels = load_data(DATA_PATH, TEXT_COLUMN, LABEL_COLUMN)
    X_train, X_test, y_train, y_test = split_data(texts, labels)
    vectorizer, X_train_tfidf, X_test_tfidf = vectorize_text(X_train, X_test)

    # 2. Train Logistic Regression model
    print("\n" + "="*60)
    print("TRAINING LOGISTIC REGRESSION MODEL")
    print("="*60)
    clf_logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf_logreg = train_model(clf_logreg, X_train_tfidf, y_train)

    # 3. Evaluate Logistic Regression
    y_pred_logreg = evaluate_model(clf_logreg, X_test_tfidf, y_test)

    # 4. ROC + AUC for Logistic Regression
    plot_roc_auc(clf_logreg, X_test_tfidf, y_test, out_path="roc_curve_logreg.png")

    # 5. Train Multinomial Naive Bayes model
    print("\n" + "="*60)
    print("TRAINING MULTINOMIAL NAIVE BAYES MODEL")
    print("="*60)
    clf_nb = MultinomialNB()
    clf_nb = train_model(clf_nb, X_train_tfidf, y_train)

    # 6. Evaluate Multinomial Naive Bayes
    y_pred_nb = evaluate_model(clf_nb, X_test_tfidf, y_test)

    # 7. ROC + AUC for Multinomial Naive Bayes
    plot_roc_auc(clf_nb, X_test_tfidf, y_test, out_path="roc_curve_nb.png")

    # 8. Save both models + vectorizer for reuse (web app, simulations, etc.)
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    save_model(vectorizer, clf_logreg, model_path=MODEL_PATH)
    save_model(vectorizer, clf_nb, model_path=NB_MODEL_PATH)

    # 9. Show some predictions
    print("\n" + "="*60)
    print("EXAMPLE PREDICTIONS (Logistic Regression)")
    print("="*60)
    show_example_predictions(clf_logreg, X_test, X_test_tfidf, y_test, y_pred_logreg)

    # 10. LIME explanation for one test email (Logistic Regression)
    sample_index = 0
    sample_text = X_test[sample_index]
    print("\n" + "="*60)
    print("LIME EXPLANATION (Logistic Regression)")
    print("="*60)
    print("Explaining this email (index 0):")
    print(sample_text[:300].replace("\n", " "))

    explain_with_lime(clf_logreg, vectorizer, sample_text)

    # 11. SHAP explanations (Logistic Regression)
    if _SHAP_AVAILABLE:
        print("\n" + "="*60)
        print("SHAP EXPLANATIONS (Logistic Regression)")
        print("="*60)
        feature_names = vectorizer.get_feature_names_out()
        explainer = build_shap_explainer(clf_logreg, X_train_tfidf, background_size=2000)

        explain_with_shap_global(
            explainer,
            clf_logreg,
            X_train_tfidf,
            feature_names,
            top_n=20,
            sample_size=2000,
        )

        explain_with_shap_local(
            explainer,
            clf_logreg,
            X_test_tfidf,
            feature_names,
            index=sample_index,
            y_test=y_test,
            top_n=10,
        )
    else:
        print("\nSHAP not installed. Skipping SHAP explanations.")

    # 12. Optional: quick demo of predict_single_email using the saved model
    print("\n" + "="*60)
    print("SINGLE EMAIL PREDICTION DEMO")
    print("="*60)
    demo_text = "Your account has been locked. Please click this link to verify your details."
    print("\nLogistic Regression predictions:")
    pred, pred_name, probs = predict_single_email(demo_text)
    print("Text:", demo_text)
    print("Predicted:", pred_name, f"({pred})")
    print("Probabilities:", probs)

    print("\nMultinomial Naive Bayes predictions:")
    pred_nb, pred_name_nb, probs_nb = predict_single_email(demo_text, model_path=NB_MODEL_PATH)
    print("Text:", demo_text)
    print("Predicted:", pred_name_nb, f"({pred_nb})")
    print("Probabilities:", probs_nb)



if __name__ == "__main__":
    main()
