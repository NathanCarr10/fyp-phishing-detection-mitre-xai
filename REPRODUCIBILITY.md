# Reproducibility Guide

So your project needs to be reproducible - that means anyone should be able to run it and get the same results. This guide explains how we do that.

## What Gets Reproduced

When you run the full pipeline, we rerun:

1. Cleaning and preparing the datasets
2. Training and evaluating the model
3. Running simulated attacks
4. Analyzing the attack results
5. Creating charts and visualizations

## Setup

**Minimum you need:**
- Python 3.8 or higher
- `pip install -r requirements.txt`

**If you want perfect reproducibility** (for your thesis submission):
1. Write down your Python version: `python --version`
2. Write down where Python is: `python -c "import sys; print(sys.executable)"`
3. Use the exact `requirements.txt` from submission time - don't upgrade packages

## How We Keep Things Consistent

We use a fixed random seed (42) in all the key scripts:
- When shuffling datasets
- When splitting into train/test
- When sampling for analysis

This way, if you run it twice with the same seed, you get identical results. The system automatically sets `PYTHONHASHSEED=42` and `FYP_RANDOM_SEED=42` to ensure everything is deterministic.

## Model Files vs. Dependencies

The trained model files (`.joblib`) are sensitive to scikit-learn versions. Important:

1. Always train and test using the same environment (`requirements.txt`)
2. If you retrain models, keep the updated model files AND document what you did
3. If your training scikit-learn version differs from testing version, you'll get a warning

Quick check your scikit-learn version:
```bash
python -c "import sklearn; print(sklearn.__version__)"
```

## Running Everything at Once

Just run this from the project folder:

```powershell
.\run_reproducible_pipeline.cmd
```

Or on Mac/Linux:

```bash
python src/reproduce_pipeline.py --seed 42
```

If you already have the datasets and just want to retrain:

```bash
python src/reproduce_pipeline.py --seed 42 --skip-build-dataset
```

## What Gets Created

After a full run, you'll have:

**Model files:**
- `models/tfidf_vectorizer.joblib` - converts text to numbers
- `models/logreg_model.joblib` - the actual phishing classifier
- `models/multinomial_nb_model.joblib` - alternative classifier

**Results:**
- `simulation_results/attacker_simulation_log.csv` - raw attack results
- `simulation_results/analysis/*.csv` - summaries and statistics
- `simulation_results/figures/*.png` - charts and visualizations
- `simulation_results/reproducibility_run_metadata.json` - run details

## Run Details Saved

Each time you run it, we save a metadata file with:

- When it started and finished (UTC time)
- Your Python version and where it's installed
- What OS you're using
- The random seed used
- Environment setup details
- Duration per pipeline stage

This metadata should be kept with dissertation appendices when reporting final experimental runs.
