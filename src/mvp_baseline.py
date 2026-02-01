"""A minimal viable product (MVP) for phishing email detection using TF-IDF and Logistic Regression."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Path to CSV file
DATA_PATH = "data/phishing_email.csv"

TEXT_COLUMN = "text_combined"
LABEL_COLUMN = "label"


def load_data(path: str, text_col: str, label_col: str):
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
        stratify=labels
    )

    print("\nTraining samples:", len(X_train))
    print("Test samples:", len(X_test))
    return X_train, X_test, y_train, y_test


def vectorize_text(X_train, X_test):
    """Fit TF-IDF on training data and transform both train and test."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=5000
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

        # Show prediction probabilities (if available)
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_test_tfidf[i])[0]
            print("Class probabilities:")
            for label, p in zip(clf.classes_, proba):
                print(f"  {label}: {p:.3f}")

        print("-" * 60)


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


if __name__ == "__main__":
    main()
