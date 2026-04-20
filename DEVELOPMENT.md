# Development Guide

This guide explains how to set up, run, and extend the project while keeping results consistent.

## Prerequisites

- Python 3.8+
- pip or conda
- Git
- Virtual environment (recommended)

## Setup Development Environment

### 1. Clone Repository

```bash
git clone <repository-url>
cd fyp-phishing-detection-mitre-xai
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n phishing-detection python=3.9
conda activate phishing-detection
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

# Optional development tools
pip install pytest pytest-cov black flake8 mypy
```

## Project Workflow

### 1. Data Preparation

```bash
python src/build_dataset.py
```

Expected output:
- `data/processed/english_dataset.csv`
- `data/processed/english_dataset_balanced.csv`

### 2. Model Training

```bash
python src/mvp_baseline.py
```

Expected outputs:
- `models/tfidf_vectorizer.joblib`
- `models/logreg_model.joblib`
- `models/multinomial_nb_model.joblib`
- ROC plot(s) and optional LIME html output

### 3. Run Adversarial Simulation

```bash
python src/attacker_sim.py
```

Expected output:
- `simulation_results/attacker_simulation_log.csv`

Simulation note:
- Attacker simulation runs at thresholds `0.5`, `0.6`, and `0.7`.
- The full threshold sweep (`0.10` to `0.95`, step `0.05`) is produced by `src/evaluate_models_rigorously.py`.

### 4. Analyze and Visualize Results

```bash
python src/analyse_simulation_results.py
python src/visualise_results.py
```

Expected outputs:
- CSV summaries in `simulation_results/analysis/`
- PNG figures in `simulation_results/figures/`

### 5. Run Rigorous Evaluation (Optional but recommended)

```bash
python src/evaluate_models_rigorously.py
```

Expected outputs:
- `evaluation_results/cv_fold_metrics.csv`
- `evaluation_results/cv_summary_metrics.csv`
- `evaluation_results/threshold_sensitivity_logreg.csv`
- `evaluation_results/calibration_logreg.csv`

### 6. Launch Interactive Dashboard

```bash
streamlit run src/app.py
```

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

## Code Quality

### Formatting and Linting

```bash
black src/ tests/
flake8 src/ tests/
```

### Type Checking

```bash
mypy src/
```

## Module Organization

### Core Modules

- `mvp_baseline.py`: model training, evaluation, ROC, optional SHAP exploration
- `xai_explainer.py`: LIME explanations with linear-weight fallback
- `attacker_sim.py`: adversarial simulation and attack generation
- `analyse_simulation_results.py`: grouped simulation metrics
- `visualise_results.py`: chart generation
- `utils.py`: shared utility functions

### Supporting Modules

- `app.py`: Streamlit dashboard
- `build_dataset.py`: dataset preparation
- `evaluate_models_rigorously.py`: CV + threshold sensitivity + calibration
- `evaluate_mitre_mapping.py`: MITRE mapping validation
- `run_error_analysis.py`: error analysis outputs

XAI reporting note:
- Final dissertation/app explainability results use LIME + linear fallback.
- SHAP remains available in baseline experimentation code for optional exploration.

## Troubleshooting

### Model files not found

```bash
python src/mvp_baseline.py
```

### LIME not installed

```bash
pip install lime
```

The system falls back to linear-weight explanations if LIME is unavailable.

## Documentation

- `README.md`: overview and quick start
- `ARCHITECTURE.md`: design and data flow
- `PROJECT_REPORT.md`: report draft
- `REPRODUCIBILITY.md`: deterministic rerun guide
- `data/README.md`: dataset notes

## Git Workflow

```bash
git checkout -b feature/my-feature
git add .
git commit -m "Your commit message"
git push origin feature/my-feature
```

---

**Last Updated**: April 2026  
**Maintainer**: Nathan (FYP Student)
