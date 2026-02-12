import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from lime.lime_text import LimeTextExplainer

# Path to your CSV file
DATA_PATH = "data/phishing_email.csv"

# Column names in the CSV
TEXT_COLUMN = "text_combined"
LABEL_COLUMN = "label"  # 1 for phishing, 0 for legitimate


def load_data(path, text_col, label_col):
    """Load the CSV file and return texts and labels."""
    print("Loading dataset from:", path)
    df = pd.read_csv(path)

    print("\nFirst few rows of the dataset:")
    print(df.head())

    print("\nColumns in the dataset:")
    print(df.columns)

    # Check that the expected columns exist
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"\nColumn names not found.\n"
            f"Current TEXT_COLUMN='{text_col}', LABEL_COLUMN='{label_col}'\n"
            f"Available columns: {list(df.columns)}\n"
            f"Edit TEXT_COLUMN and LABEL_COLUMN at the top of the file."
        )

    # Drop rows where text or label is missing
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
        print(X_test[i][:200].replace("\n", " "))  # first 200 characters
        print("True label:   ", y_test[i])
        print("Predicted:    ", y_pred[i])

        # Show prediction probabilities
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_test_tfidf[i])[0]
            print("Class probabilities:")
            for label, p in zip(clf.classes_, proba):
                print(f"  {label}: {p:.3f}")

        print("-" * 60)


def explain_with_lime(clf, vectorizer, text_sample):
    """Use LIME to explain one email."""
    # Function that LIME will call:
    # input: list of texts
    # output: list of probability vectors
    def predict_proba(text_list):
        X = vectorizer.transform(text_list)
        return clf.predict_proba(X)
    
    class_names = [str(c) for c in clf.classes_]

    explainer = LimeTextExplainer(class_names=class_names)

    # Explain this one email
    exp = explainer.explain_instance(
        text_sample,
        predict_proba,
        num_features=10  # show top 10 important words
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


def main():
    # 1. Load data
    texts, labels = load_data(DATA_PATH, TEXT_COLUMN, LABEL_COLUMN)

    # 2. Split into train and test
    X_train, X_test, y_train, y_test = split_data(texts, labels)

    # 3. Vectorize text with TF-IDF
    vectorizer, X_train_tfidf, X_test_tfidf = vectorize_text(X_train, X_test)

    # 4. Train classifier
    clf = train_model(X_train_tfidf, y_train)

    # 5. Evaluate model
    y_pred = evaluate_model(clf, X_test_tfidf, y_test)

    # 6. Show example predictions
    show_example_predictions(clf, X_test, X_test_tfidf, y_test, y_pred)

    # 7. Explain one email with LIME
    sample_index = 0 # Change this index to explain a different email
    sample_text = X_test[sample_index]
    print("\nExplaining this email with LIME:")
    print(sample_text[:300].replace("\n", " "))

    explain_with_lime(clf, vectorizer, sample_text)


if __name__ == "__main__":
    main()
