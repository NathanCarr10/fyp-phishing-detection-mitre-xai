# Reproducibility Guide

This guide explains how to rerun the project and get consistent results.

## What Gets Reproduced

A full run reproduces:

1. Dataset preparation
2. Model training and evaluation
3. Attacker simulation
4. Simulation analysis
5. Figure generation

## Environment Setup

Minimum:
- Python 3.8+
- `pip install -r requirements.txt`

For strict reproducibility:

1. Record Python version: `python --version`
2. Record Python executable: `python -c "import sys; print(sys.executable)"`
3. Use the pinned dependency versions from submission-time `requirements.txt`

## Determinism Controls

The project uses fixed seeds (`42`) in key steps (splits, shuffles, sampling), and the reproducibility runner exports:

- `PYTHONHASHSEED=42`
- `FYP_RANDOM_SEED=42`

## Running the Full Pipeline

From project root:

```powershell
.\run_reproducible_pipeline.cmd
```

Or directly:

```bash
python src/reproduce_pipeline.py --seed 42
```

If processed datasets already exist:

```bash
python src/reproduce_pipeline.py --seed 42 --skip-build-dataset
```

## Threshold Scope Clarification

- `src/attacker_sim.py` runs thresholds `0.5`, `0.6`, `0.7` and writes `simulation_results/attacker_simulation_log.csv`.
- `src/evaluate_models_rigorously.py` performs a separate threshold sensitivity sweep from `0.10` to `0.95` (step `0.05`) and writes `evaluation_results/threshold_sensitivity_logreg.csv`.

## Explainability Scope Clarification

- Final dissertation/app explainability reporting is based on LIME + linear fallback.
- SHAP support exists in baseline experimentation code (`src/mvp_baseline.py`) as optional exploratory analysis.

## Key Outputs

Model artifacts:
- `models/tfidf_vectorizer.joblib`
- `models/logreg_model.joblib`
- `models/multinomial_nb_model.joblib`

Simulation artifacts:
- `simulation_results/attacker_simulation_log.csv`
- `simulation_results/analysis/*.csv`
- `simulation_results/figures/*.png`

Rigorous evaluation artifacts:
- `evaluation_results/cv_fold_metrics.csv`
- `evaluation_results/cv_summary_metrics.csv`
- `evaluation_results/threshold_sensitivity_logreg.csv`
- `evaluation_results/calibration_logreg.csv`

Run metadata:
- `simulation_results/reproducibility_run_metadata.json`

## Metadata Saved Per Run

Each reproducible run records:

- UTC start and finish times
- Python executable and version
- OS/platform info
- Seed and reproducibility environment variables
- Per-stage durations

Keep this metadata with dissertation appendices for reported final runs.
