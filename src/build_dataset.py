# src/build_dataset.py
#
# Builds a smaller, easier-to-explain dataset:
# - Enron = legit (0)
# - Spam/Phishing sources = phishing (1)
# Saves: data/processed/english_dataset.csv with columns: text, label

import os
import pandas as pd


RAW_DIR = "data/raw"
OUT_PATH = "data/processed/english_dataset.csv"

# Choose which files to use and what label they represent
SOURCES = [
    ("Enron.csv", 0),           # legit
    ("SpamAssasin.csv", 1),     # phishing/spam
    ("Nazario.csv", 1),         # phishing
    ("Nigerian_Fraud.csv", 1),  # phishing
]

TEXT_COLUMN_CANDIDATES = [
    "text", "email", "body", "content", "message", "email_text",
    "text_combined", "mail", "data"
]


def pick_text_column(df: pd.DataFrame) -> str:
    """Try to find the best column that contains email text."""
    cols = list(df.columns)

    # 1) Look for common names
    for c in TEXT_COLUMN_CANDIDATES:
        if c in cols:
            return c

    # 2) Look for any column containing 'text' or 'body' in the name
    for c in cols:
        name = str(c).lower()
        if "text" in name or "body" in name or "message" in name or "content" in name:
            return c

    # 3) Fallback: first column that looks like text (object/string dtype)
    for c in cols:
        if df[c].dtype == "object":
            return c

    # Error if no suitable column found
    raise ValueError(f"Could not find a text column. Columns are: {cols}")


def load_source(filename: str, label_value: int) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, filename)
    print(f"\nLoading: {path}")

    df = pd.read_csv(path)

    text_col = pick_text_column(df)
    print(f"Using text column: {text_col}")

    out = pd.DataFrame({
        "text": df[text_col].astype(str),
        "label": label_value
    })

    # Drop empty text
    out["text"] = out["text"].str.strip()
    out = out[out["text"].str.len() > 0]

    print(f"Rows kept: {len(out)}")
    return out


def main():
    all_parts = []

    for filename, label_value in SOURCES:
        part = load_source(filename, label_value)
        all_parts.append(part)

    combined = pd.concat(all_parts, ignore_index=True)

    # Shuffle rows so train/test is mixed
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # Ensure output folder exists
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    combined.to_csv(OUT_PATH, index=False)
    print(f"\nSaved combined dataset to: {OUT_PATH}")
    print(combined["label"].value_counts())


if __name__ == "__main__":
    main()
