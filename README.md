# Phishing Detection with MITRE ATT&CK Mapping and Explainable AI

> Final Year Project for Computing in Software Development at Atlantic Technological University, Galway

## Overview

This project builds an email phishing detection system that can not only classify emails as legitimate or phishing, but also explain *why* it made that choice. It maps detected threats to the MITRE ATT&CK framework and uses techniques like LIME and SHAP to give explainable predictions.

Screencast (5 min): [View or download the project video](https://raw.githubusercontent.com/NathanCarr10/fyp-phishing-detection-mitre-xai/main/Dissertation%20%2B%20Screencast/Final%20Year%20Project%20Screencast%20(5%20min).mp4)
Full Screencast (YouTube): [Watch the extended video](https://www.youtube.com/watch?v=CVL3C_xHcic)

### Key Features

- 🔍 **Phishing Detection**: Classifies emails using Logistic Regression with TF-IDF
- 🛡️ **MITRE ATT&CK Mapping**: Automatically maps detected phishing techniques to five industry threat categories
- 🤖 **Explainable Results**: Shows exactly which words pushed the model toward phishing or legitimate
- ⚔️ **Adversarial Testing**: Simulates realistic attacker scenarios across four attack rules and three thresholds
- 📊 **Detailed Analysis**: Cross-validation, threshold tuning, calibration, and error analysis
- 🎨 **Interactive Dashboard**: Streamlit app for classifying emails and viewing explanations in real-time

## Quick Start

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NathanCarr10/fyp-phishing-detection-mitre-xai
   cd fyp-phishing-detection-mitre-xai
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Package versions are locked to ensure everything works together correctly.

3. **Download and prepare datasets:**
   ```bash
   # See data/README.md for detailed instructions
   # Place CSV files in data/raw/ directory
   ```

### Reproducible One-Command Run

To rerun the full experiment pipeline with fixed reproducibility settings:

```powershell
.\run_reproducible_pipeline.cmd
```

Equivalent Python command:

```bash
python src/reproduce_pipeline.py --seed 42
```

This executes dataset build, training, simulation, analysis, and visualisation in order, then writes run metadata to `simulation_results/reproducibility_run_metadata.json`.

For full reproducibility details, see `REPRODUCIBILITY.md`.

### Running the Project

Here is the basic workflow:

#### 1. Train the Model

```bash
python src/mvp_baseline.py
```

This loads all the email datasets, trains both models, and saves them to the `models/` folder. Metrics including accuracy, precision, recall, F1 score, and AUC are printed to the terminal along with ROC curves for both classifiers.

#### 1b. Run Rigorous Testing (Cross-Validation, Confidence Intervals, Calibration)

```bash
python src/evaluate_models_rigorously.py
```

This tests the model more thoroughly by:
- Running repeated stratified cross-validation (5 folds, 3 repeats, 15 runs per model)
- Computing 95% confidence intervals across fold scores
- Testing decision thresholds from 0.10 to 0.95 to find the optimal operating point
- Checking calibration using Brier score and Expected Calibration Error (ECE)

#### 1c. Check MITRE Mapping Quality

```bash
python src/evaluate_mitre_mapping.py
```

Formally evaluates the MITRE mapping against a manually labelled validation subset, computing primary-label accuracy, macro F1, exact match rate, and micro precision, recall, and F1.

#### 1d. Analyse Model Errors

```bash
python src/run_error_analysis.py
```

Generates reports on:
- False positives (legitimate emails marked as phishing)
- False negatives (phishing emails marked as legitimate)
- The fifty most confidently wrong predictions in each direction
- The most common vocabulary appearing in each error type

#### 2. Run Adversarial Attacks

```bash
python src/attacker_sim.py
```

Simulates realistic attacks to test model robustness:
- Applies four attack rules individually and in pairs (16 combinations per base email)
- Tests across three classification thresholds (0.5, 0.6, 0.7)
- Records which attacks were caught and which bypassed detection
- Logs LIME explanations for every bypass case

#### 3. Analyse Simulation Results

```bash
python src/analyse_simulation_results.py
```

This will:
- Compute confusion matrix metrics
- Generate summary tables grouped by threshold, attack type, rule chain, and MITRE technique
- Output seven summary CSVs to `simulation_results/analysis/`

#### 4. Visualise Results

```bash
python src/visualise_results.py
```

This will:
- Generate charts from the simulation analysis CSVs
- Create threshold analysis plots
- Plot detection rates by attack type and rule chain
- Generate MITRE technique effectiveness charts
- Output to `simulation_results/figures/`

#### 5. Run the Dashboard

```bash
python -m streamlit run src/app.py
```

Or on Windows:

```powershell
.\run_app.cmd
```

This starts a web interface where you can:
- Paste in email text or upload a .eml file to classify it
- Adjust the classification threshold (0.10 to 0.95)
- Switch between Logistic Regression and Naive Bayes
- Toggle LIME and SHAP explanation methods
- View confidence scores and MITRE technique mapping
- See which words drove the classification decision

## Project Structure

```
fyp-phishing-detection-mitre-xai/
├── src/
│   ├── build_dataset.py                  # Dataset construction and balancing
│   ├── mvp_baseline.py                   # Model training and evaluation
│   ├── models.py                         # Reusable model factory and helpers
│   ├── xai_explainer.py                  # SHAP, LIME, and linear weight explanations
│   ├── attacker_sim.py                   # Adversarial simulation and MITRE mapping
│   ├── analyse_simulation_results.py     # Simulation log processing
│   ├── visualise_results.py              # Chart generation
│   ├── compare_models.py                 # Side-by-side model comparison
│   ├── evaluate_models_rigorously.py     # Cross-validation, threshold sensitivity, calibration
│   ├── evaluate_mitre_mapping.py         # Formal MITRE mapping evaluation
│   ├── run_error_analysis.py             # False positive and negative analysis
│   ├── reproduce_pipeline.py             # Pipeline runner with seed control
│   ├── email_ingestion.py                # .eml file parsing
│   ├── utils.py                          # Shared constants and helper functions
│   └── app.py                            # Streamlit dashboard
│
├── data/
│   ├── raw/                              # Original datasets (not committed)
│   ├── processed/
│   │   └── english_dataset.csv           # Processed training dataset
│   └── README.md                         # Dataset documentation
│
├── models/
│   ├── tfidf_vectorizer.joblib           # Trained TF-IDF vectoriser
│   ├── logreg_model.joblib               # Trained Logistic Regression model
│   └── multinomial_nb_model.joblib       # Trained Naive Bayes model
│
├── simulation_results/
│   ├── attacker_simulation_log.csv       # Raw simulation logs
│   ├── analysis/                         # Summary statistics CSVs
│   └── figures/                          # Generated visualisations
│
├── evaluation_results/                   # Cross-validation, MITRE eval, and error analysis outputs
│
├── tests/
│   ├── conftest.py                       # Shared fixtures and custom pytest markers
│   ├── test_utils.py                     # Tests for utility functions
│   ├── test_xai_explainer.py             # Tests for explanation methods
│   ├── test_email_ingestion.py           # Tests for .eml parsing
│   └── test_stage_scripts_helpers.py     # Tests for evaluation and analysis helpers
│
├── .github/
│   └── workflows/
│       └── ci.yml                        # GitHub Actions CI workflow
│
├── ARCHITECTURE.md                       # System design documentation
├── REPRODUCIBILITY.md                    # Reproducibility protocol and metadata guidance
├── DEVELOPMENT.md                        # Development notes
├── PROJECT_REPORT.md                     # Research report
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
├── run_reproducible_pipeline.cmd         # One-command reproducible pipeline launcher
└── .gitignore
```

## Technical Details

### Model Architecture

**Primary Classifier**: Logistic Regression with `class_weight="balanced"`
**Comparison Classifier**: Multinomial Naive Bayes

**Feature Extraction**: TF-IDF Vectorisation
- 5,000 maximum features
- Lowercase text normalisation
- English stopword removal
- Fitted on training set only to prevent data leakage

**Training Data**: 21,402 emails balanced equally across both classes
- Legitimate emails: Enron corpus
- Phishing emails: SpamAssassin, Nazario collection, Nigerian Fraud emails

### Explainability

The system supports three explanation methods applied in priority order:

1. **SHAP** (`shap.LinearExplainer`)
   - Global feature importance across the full dataset
   - Built from a 200-email background sample and cached after first call
   - Reveals which vocabulary the model relies on most overall

2. **LIME** (Local Interpretable Model-agnostic Explanations)
   - Per-email local explanations
   - Generates perturbations of the input and fits a local interpretable model
   - Returns top contributing words with weights

3. **Linear Weight Analysis** (fallback)
   - Computes feature importance directly from TF-IDF value × Logistic Regression coefficient
   - Runs in under 1ms with no additional libraries required
   - Used automatically when LIME and SHAP are unavailable

All three methods return results in the same structured format.

### MITRE ATT&CK Mapping

Pattern and keyword scoring system covering five techniques:

| Technique | Description |
|-----------|-------------|
| T1566.002 | Phishing: Link |
| T1566.001 | Phishing: Attachment |
| T1598.003 | Spearphishing: Social Platform Lookalike |
| T1598.001 | Spearphishing: Attachment |
| T1598.002 | Spearphishing: Shortened URL |

URL pattern matches score 1 point each and phishing keyword matches score 2 points each. The email is assigned to the highest-scoring technique, with T1566.001 used as a fallback when nothing matches. Mapping accuracy was formally evaluated against a manually labelled validation subset — see `src/evaluate_mitre_mapping.py`.

### Adversarial Simulation

Simulates attacker vs. defender scenarios across four attack rules:

| Rule | Description |
|------|-------------|
| `urgency` | Adds time pressure language |
| `spoof_bank` | Spoofed bank security notice |
| `spoof_revenue` | Spoofed tax authority message |
| `fake_link` | Injects a malicious URL |

Rules are applied individually (4 single-rule attacks) and in pairs (12 two-rule combinations) per base email, across three classification thresholds (0.5, 0.6, 0.7). LIME explanations are automatically logged for every bypass case.

## Performance Results

### Model Evaluation (held-out test set, 4,281 emails)

| Metric | Logistic Regression | Naive Bayes |
|--------|-------------------|-------------|
| Accuracy | 93.76% | 90.89% |
| Precision | 0.94 | 0.91 |
| Recall | 0.94 | 0.91 |
| F1 Score | 0.94 | 0.91 |
| AUC-ROC | 0.9808 | 0.9642 |

### Cross-Validation (5 folds × 3 repeats = 15 runs)

| Metric | LR Mean | LR 95% CI |
|--------|---------|-----------|
| Accuracy | 0.938 | [0.933, 0.941] |
| AUC | 0.980 | [0.978, 0.982] |

### Attacker Simulation

| Threshold | Detection Rate | Bypass Rate |
|-----------|---------------|-------------|
| 0.5 | 96.9% | 3.1% |
| 0.6 | 84.4% | 15.6% |
| 0.7 | 57.8% | 42.2% |

### MITRE Mapping Evaluation

| Metric | Score |
|--------|-------|
| Primary-label Accuracy | 93.3% |
| Macro F1 | 0.931 |
| Multi-label Micro Recall | 0.941 |

## Dependencies

| Library | Purpose |
|---------|---------|
| pandas | Data processing |
| scikit-learn | ML model training and TF-IDF |
| lime | Local explanations |
| shap | Global feature importance |
| matplotlib | Visualisation |
| streamlit | Interactive dashboard |
| joblib | Model persistence |

See `requirements.txt` for specific versions.

## Continuous Integration

A GitHub Actions workflow runs the test suite automatically on every push:

- `.github/workflows/ci.yml`

Local equivalent:

```bash
python -m pytest tests/ -v
```

## Usage Examples

### Classify a Single Email

```python
from src.utils import load_model
from src.xai_explainer import explain_email

vectorizer, clf = load_model()
email_text = "URGENT: Your account has been compromised. Click here to verify."

explanation = explain_email(
    email_text,
    num_features=10,
    threshold=0.5,
    use_lime=True
)

print(f"Prediction: {'Phishing' if explanation['is_phishing'] else 'Legitimate'}")
print(f"Phishing Probability: {explanation['phishing_probability']:.3f}")
for feature in explanation['top_features']:
    print(f"  {feature['term']}: {feature['weight']:+.4f}")
```

### Map to MITRE ATT&CK

```python
from src.attacker_sim import mitre_mapping

email = "Click here to verify your account: http://secure-verification-example.com/login"
technique = mitre_mapping(email)
print(f"MITRE Technique: {technique}")
# Output: T1566.002 - Phishing: Link
```

## Limitations

- The model is trained on datasets from 1999 to 2002. The SHAP analysis confirms this directly — several top-ranked global features are Enron-specific terms rather than genuine phishing vocabulary. Results should be treated as a proof of concept rather than a production-ready system.
- The attacker simulation uses four predefined rules. A real attacker can adapt in ways that fixed rules cannot replicate. Bypass rates should be considered a floor rather than a ceiling.
- LIME explanations can vary between runs due to random perturbations. They should be treated as a guide rather than a definitive account of the model's reasoning.

## Future Work

- Compare BERT and DistilBERT on the same evaluation pipeline
- Implement gradient-based adversarial attacks
- Add URL analysis and sender metadata features (SPF, DKIM)
- Retrain on a modern dataset including LLM-generated phishing emails
- Deploy as a REST API for real-time email filtering

## References

- **LIME** (Ribeiro et al., 2016): "Why Should I Trust You?"
- **SHAP** (Lundberg & Lee, 2017): A unified approach to interpreting model predictions
- **scikit-learn**: Machine learning in Python
- **MITRE ATT&CK**: Adversary tactics, techniques, and procedures framework

## License

This project is submitted as coursework for ATU Galway. Please contact the author for usage or distribution permissions.

## Contact

**Author**: Nathan Carr (G00410214)
**Degree**: BSc Computing in Software Development
**Institution**: Atlantic Technological University, Galway
**Submission Date**: April 2026

## Acknowledgements

- ATU Galway for project supervision
- Enron, SpamAssassin, Nazario, and Nigerian Fraud dataset contributors
- The open-source LIME, SHAP, and scikit-learn communities
- MITRE ATT&CK framework creators

---

**Last Updated**: April 2026
**Status**: Complete
