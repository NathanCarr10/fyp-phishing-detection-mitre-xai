# fyp-phishing-detection-mitre-xai
My Final Year Project for Computing in Software Development in ATU Galway: AI-powered phishing email detection system with explainable AI (LIME &amp; SHAP), simple attacker simulation, and MITRE ATT&amp;CK / ATLAS mapping.

## How to run
1. Install dependencies:
   pip install -r requirements.txt

2. Run the baseline model:
   python src/mvp_baseline.py

## Dataset

The email datasets (CSV files) are **not** stored in this repository because they are large.

They come from a Kaggle phishing email dataset and are stored locally in the `data/` folder.

To run the project, download the dataset from Kaggle and place the CSV files in:

`data/`
- phishing_email.csv
- CEAS_08.csv
- Enron.csv
- ...

(The model currently uses `phishing_email.csv` for the MVP.)
