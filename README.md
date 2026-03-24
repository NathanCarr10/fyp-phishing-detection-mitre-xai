# Phishing Detection with MITRE ATT&CK Mapping and Explainable AI

> Final Year Project for Computing in Software Development at Atlantic Technological University, Galway

## Overview

This project implements an **AI-powered email phishing detection system** with integrated **MITRE ATT&CK threat mapping** and **explainable AI (XAI)** explanations. The system classifies emails as legitimate or phishing, maps detected threats to the MITRE ATT&CK framework, and provides interpretable explanations for predictions using LIME and linear model weights.

### Key Features

- 🔍 **Phishing Detection**: Binary classification using Logistic Regression with TF-IDF feature extraction
- 🛡️ **MITRE ATT&CK Mapping**: Automatic mapping of detected phishing techniques to MITRE ATT&CK categories
- 🤖 **Explainable AI**: LIME-based and linear-weight explanations for model predictions
- ⚔️ **Adversarial Simulation**: Attacker vs. defender simulation with multi-rule attack chaining
- 📊 **Comprehensive Analysis**: Threshold analysis, attack effectiveness metrics, and visualization dashboard
- 🎨 **Interactive UI**: Streamlit-based dashboard for real-time email classification and explanation

## Quick Start

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd fyp-phishing-detection-mitre-xai
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and prepare datasets:**
   ```bash
   # See data/README.md for detailed instructions
   # Place CSV files in data/raw/ directory
   ```

### Running the Project

#### 1. Train the Baseline Model

```bash
python src/mvp_baseline.py
```

This will:
- Load and preprocess email datasets
- Train a TF-IDF + Logistic Regression model
- Save the trained model and vectorizer to `models/`
- Display evaluation metrics (accuracy, precision, recall, F1, AUC)
- Generate ROC curve visualization

#### 2. Run the Attacker Simulation

```bash
python src/attacker_sim.py
```

This will:
- Simulate attacks using predefined phishing rules
- Test multi-rule attack chaining
- Generate simulation logs with results
- Log XAI explanations for attack successes
- Output: `simulation_results/attacker_simulation_log.csv`

#### 3. Analyze Simulation Results

```bash
python src/analyse_simulation_results.py
```

This will:
- Compute confusion matrix metrics
- Generate summary tables grouped by threshold, attack type, rule chain, and MITRE technique
- Output CSVs to `simulation_results/analysis/`

#### 4. Visualize Results

```bash
python src/visualise_results.py
```

This will:
- Generate publication-ready charts
- Create threshold analysis plots
- Plot detection rates by attack type
- Generate MITRE technique effectiveness charts
- Output: `simulation_results/figures/`

#### 5. Launch Interactive Dashboard

```bash
streamlit run src/app.py
```

This will:
- Start a local web server (http://localhost:8501)
- Provide interactive interface for:
  - Pasting email text for classification
  - Adjusting classification threshold
  - Viewing phishing probability scores
  - Inspecting MITRE ATT&CK mappings
  - Examining XAI explanations
  - Browsing simulation analysis figures

## Project Structure

```
fyp-phishing-detection-mitre-xai/
├── src/
│   ├── mvp_baseline.py              # Model training and evaluation
│   ├── xai_explainer.py             # XAI explanation module (LIME + weights)
│   ├── attacker_sim.py              # Adversarial simulation engine
│   ├── analyse_simulation_results.py # Analysis pipeline
│   ├── visualise_results.py          # Visualization generation
│   ├── build_dataset.py              # Dataset preprocessing
│   ├── app.py                        # Streamlit dashboard
│   └── utils.py                      # Shared utility functions
│
├── data/
│   ├── raw/                          # Original datasets (not in repo)
│   ├── processed/
│   │   └── english_dataset.csv       # Processed training dataset
│   └── README.md                     # Dataset documentation
│
├── models/
│   ├── tfidf_vectorizer.joblib       # Trained TF-IDF vectorizer
│   └── logreg_model.joblib           # Trained Logistic Regression model
│
├── simulation_results/
│   ├── attacker_simulation_log.csv   # Raw simulation logs
│   ├── analysis/                     # Summary statistics CSVs
│   └── figures/                      # Generated visualizations
│
├── tests/
│   ├── test_mvp_baseline.py
│   ├── test_xai_explainer.py
│   ├── test_attacker_sim.py
│   └── conftest.py
│
├── ARCHITECTURE.md                   # System design documentation
├── PROJECT_REPORT.md                 # Research report
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── .gitignore
```

## Technical Details

### Model Architecture

**Classifier**: Logistic Regression with balanced class weights
- **Feature Extraction**: TF-IDF Vectorization
  - 5000 maximum features
  - Lowercase text normalization
  - English stopwords removal
- **Training Data**: Combined dataset from multiple sources
  - Legitimate emails: Enron corpus
  - Phishing emails: Kaggle phishing datasets, Nazario collection, Nigerian fraud emails

### Explainability

The system supports two explanation methods:

1. **LIME (Local Interpretable Model-agnostic Explanations)**
   - Generates local linear approximations of model decisions
   - Returns top features (words) contributing to phishing score
   - Provides intuitive explanation with feature weights

2. **Linear Weights (Fallback)**
   - Uses TF-IDF weight × model coefficient for each feature
   - Simpler, faster explanation without additional computation
   - Automatically used if LIME unavailable

### MITRE ATT&CK Mapping

Simple pattern-based mapping to detect common phishing techniques:
- **T1566.002 (Phishing: Link)**: Detects URLs in email content
- **T1566.001 (Phishing: Attachment/Generic)**: Default category

*Note: Mapping can be extended to cover more MITRE techniques. See ARCHITECTURE.md.*

### Adversarial Simulation

Simulates attacker vs. defender scenarios:

**Attack Rules:**
- `urgency`: Adds time pressure language
- `spoof_bank`: Spoofed bank security notice
- `spoof_revenue`: Spoofed tax authority
- `fake_link`: Injects malicious link

**Attack Variants:**
- Single-rule attacks (4 variants per base email)
- Multi-rule chained attacks (4 × 3 = 12 two-rule combinations)

**Metrics Computed:**
- TP, FP, TN, FN (confusion matrix)
- Detection Rate (TPR)
- Bypass Rate (FNR)
- Precision, Recall, F1 Score
- Accuracy

## Performance Results

### Model Evaluation (on test set)

| Metric | Score |
|--------|-------|
| Accuracy | ~95% |
| Precision | ~94% |
| Recall | ~96% |
| F1 Score | ~95% |
| AUC | ~0.98 |

*Exact figures depend on dataset and train/test split. See simulation_results/ for detailed analysis.*

### Simulation Analysis

The adversarial simulation tests model robustness against:
- Single rule attacks (4 rule types)
- Multi-rule attacks (12 combinations)
- Threshold sensitivity (0.1 to 0.95)
- Performance summarized by:
  - Attack type
  - Rule chain
  - MITRE technique
  - Classification threshold

## Dependencies

| Library | Purpose |
|---------|---------|
| pandas | Data processing |
| scikit-learn | ML model training & TF-IDF |
| lime | Local explanations |
| shap | Global model explanations |
| matplotlib | Visualization |
| streamlit | Interactive dashboard |
| joblib | Model persistence |

See `requirements.txt` for specific versions.

## Usage Examples

### Classify a Single Email

```python
from src.mvp_baseline import load_model
from src.xai_explainer import explain_email

vectorizer, clf = load_model()
email_text = "URGENT: Your account has been compromised. Click here..."

# Get explanation
explanation = explain_email(
    email_text,
    num_features=10,
    threshold=0.5,
    use_lime=True
)

print(f"Prediction: {explanation['pred_label']}")
print(f"Phishing Probability: {explanation['phishing_probability']:.3f}")
for feature in explanation['top_features']:
    print(f"  {feature['term']}: {feature['weight']:+.4f}")
```

### Map to MITRE ATT&CK

```python
from src.attacker_sim import mitre_mapping

email = "Click here to verify your account: http://example.com"
technique = mitre_mapping(email)
print(f"MITRE Technique: {technique}")
# Output: T1566.002 - Phishing: Link
```

### Generate Simulation Report

See `src/analyse_simulation_results.py` for detailed analysis pipeline.

## References & Data Sources

### Datasets

- **Enron Email Corpus**: Public email archive from bankrupt company
- **Kaggle Phishing Dataset**: https://www.kaggle.com/datasets/...
- **Nazario Phishing Corpus**: Phishing email collection
- **Nigerian Fraud Emails**: Common fraud email patterns

### Libraries & Frameworks

- **LIME** (Ribeiro et al., 2016): "Why Should I Trust You?"
- **SHAP** (Lundberg & Lee, 2017): Unified approach to interpreting model predictions
- **scikit-learn**: Machine learning toolkit
- **MITRE ATT&CK**: Adversary tactics, techniques, and procedures framework

## Limitations & Future Work

### Current Limitations

- MITRE mapping is pattern-based (limited to 2 techniques)
- Simulation uses simple rule-based attacks (not adversarial ML)
- Model trained on public datasets (may not generalize to enterprise emails)
- No temporal dynamics or email thread context

### Future Enhancements

- [ ] Extend MITRE mapping to cover more techniques (T1598, T1594, etc.)
- [ ] Implement gradient-based adversarial attacks
- [ ] Add ensemble methods (Random Forest, Gradient Boosting)
- [ ] Integrate with email systems for real-time detection
- [ ] Add deep learning models (BERT, RoBERTa)
- [ ] Implement multi-label classification for multiple attack vectors
- [ ] Add email metadata features (sender reputation, header analysis)
- [ ] Cross-validation and hyperparameter optimization studies

## Development

### Running Tests

```bash
python -m pytest tests/ -v
```

### Code Style

This project follows PEP 8 guidelines. All functions include docstrings in Google style.

### Contributing

For internal development:
1. Create a feature branch
2. Make changes with clear commit messages
3. Add tests for new functionality
4. Ensure all tests pass

## License

This project is submitted as coursework for ATU Galway. Please contact the author for usage/distribution permissions.

## Contact & Author

**Author**: Nathan (Student ID: G00410214)  
**Degree**: BSc Computing in Software Development  
**Institution**: Atlantic Technological University, Galway  
**Submission Date**: April 2026

## Acknowledgments

- ATU Galway for project supervision
- Kaggle for public datasets
- Open-source ML/XAI communities
- MITRE ATT&CK framework creators

---

**Last Updated**: March 2026  
**Status**: In progress
