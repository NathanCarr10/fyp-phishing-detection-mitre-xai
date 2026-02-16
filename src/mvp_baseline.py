import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from lime.lime_text import LimeTextExplainer
import shap

# Paths and column names
DATA_PATH = "data/processed/english_dataset.csv"
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"


# label names for printing and explanations
LABEL_MAP = {
    0: "legit",
    1: "phishing",
}


def load_data(path, text_col, label_col):
    """Load the CSV file and return texts and labels."""
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


def train_model(X_train_tfidf, y_train):
    """Train Logistic Regression classifier."""
    clf = LogisticRegression(max_iter=1000)

    print("\nTraining Logistic Regression classifier...")
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


def main():
    texts, labels = load_data(DATA_PATH, TEXT_COLUMN, LABEL_COLUMN)
    X_train, X_test, y_train, y_test = split_data(texts, labels)
    vectorizer, X_train_tfidf, X_test_tfidf = vectorize_text(X_train, X_test)
    clf = train_model(X_train_tfidf, y_train)

    y_pred = evaluate_model(clf, X_test_tfidf, y_test)
    show_example_predictions(clf, X_test, X_test_tfidf, y_test, y_pred)

    sample_index = 0
    sample_text = X_test[sample_index]
    print("\nExplaining this email (index 0) with LIME:")
    print(sample_text[:300].replace("\n", " "))

    explain_with_lime(clf, vectorizer, sample_text)

    feature_names = vectorizer.get_feature_names_out()

    explainer = build_shap_explainer(clf, X_train_tfidf, background_size=2000)

    explain_with_shap_global(explainer, clf, X_train_tfidf, feature_names, top_n=20, sample_size=2000)

    explain_with_shap_local(explainer, clf, X_test_tfidf, feature_names, index=sample_index, y_test=y_test, top_n=10)


if __name__ == "__main__":
    main()
