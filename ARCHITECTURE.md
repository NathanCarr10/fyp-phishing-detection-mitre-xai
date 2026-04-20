# System Architecture & Design

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Data Pipeline](#data-pipeline)
3. [Model Components](#model-components)
4. [XAI Explanation Methods](#xai-explanation-methods)
5. [MITRE Mapping Strategy](#mitre-mapping-strategy)
6. [Adversarial Simulation](#adversarial-simulation)
7. [Module Dependencies](#module-dependencies)
8. [Configuration & Constants](#configuration--constants)

---

## High-Level Overview

The system is split into three main parts:

```
Training Pipeline               Explanation Pipeline        Simulation Pipeline
─────────────────             ──────────────────          ──────────────────
build_dataset.py              xai_explainer.py            attacker_sim.py
        ↓                              ↓                            ↓
mvp_baseline.py ───────────→ TF-IDF + LogReg ←──────── attacker_sim.py
        ↓                              ↓                            ↓
models/ (saved)          Explanations (LIME,        simulation_results/
                         weights)
                                ↓
                          analyse_simulation_results.py
                                ↓
                          simulation_results/analysis/
                                ↓
                          visualise_results.py
                                ↓
                          simulation_results/figures/
                                ↓
                          app.py (Streamlit Dashboard)
```

In the actual implementation, we also added:

- `evaluate_models_rigorously.py` for testing the model thoroughly with cross-validation and confidence intervals
- `evaluate_mitre_mapping.py` to check how well our MITRE mapping works against labeled examples
- `run_error_analysis.py` to find and analyze false positives and false negatives
- `.github/workflows/ci.yml` to automatically run tests when code is pushed

---

## Data Pipeline

### Stage 1: Get the Data

**What we use:**
- CSV files with emails from Kaggle and academic sources
- Each file has the email text and a label (legitimate or phishing)
- All files are in `data/raw/`

### Stage 2: Clean the Data (build_dataset.py)

The preprocessing steps are:

```
Raw CSV Files
     ↓
[Find the text column] → Automatically figure out which column has the email
     ↓
[Assign labels] → Mark as 0 (legitimate) or 1 (phishing)
     ↓
[Clean up] → Remove empty or whitespace-only entries
     ↓
[Combine] → Merge all the datasets together
     ↓
[Shuffle] → Randomize the order (using seed=42 for consistency)
     ↓
english_dataset.csv (saved in data/processed/)
```

**Output:** `data/processed/english_dataset.csv`

**What's in it:**
- `text`: The email content
- `label`: 0 for legitimate, 1 for phishing

### Stage 3: Split Into Train and Test

```
english_dataset.csv
        ↓
[Split 80/20] → 80% for training, 20% for testing (seed=42)
        ↓
    Training Data        Test Data
```

**Why stratified split?** It keeps roughly the same mix of phishing vs legitimate in both train and test sets.

---

## Model Components

### 1. Feature Extraction: TF-IDF

**What it does:**
Converts email text into numbers that the model can understand. TF-IDF gives higher weight to words that are:
- Frequent in the email (Term Frequency)
- Rare across all emails (Inverse Document Frequency)

This way, common words like "the" don't mess up the detection, but important phishing indicators get noticed.

**Settings:**
```python
TfidfVectorizer(
    lowercase=True,           # Treat "CLICK" and "click" the same
    stop_words='english',     # Ignore common words (the, and, is, etc)
    max_features=5000,        # Use the 5000 most common words
)
```

### 2. Classification: Logistic Regression

**What it does:**
A simple but powerful model that learns patterns from the TF-IDF features and gives a probability score (0 to 1) for whether an email is phishing.

**Why this model?**
- It's interpretable - we can see which words the model thinks are suspicious
- It's fast - can classify hundreds of emails per second
- It gives us probability scores - we can adjust how strict we want to be
- It's a solid baseline that works well for text classification

**Settings:**
```python
LogisticRegression(
    max_iter=1000,           # Let it train until it converges
    class_weight='balanced', # Handle imbalanced data
)
```

---

## XAI Explanation Methods

### Method 1: LIME (Local Interpretable Model-Agnostic Explanations)

**Location:** `src/xai_explainer.py` → `_explain_with_lime()`

**Algorithm:**
1. Take target email text
2. Generate perturbed variations (random word deletions)
3. Get model predictions for variations
4. Fit local linear model to approximate decision boundary
5. Extract feature weights from linear model
6. Return top K features (words) with highest impact

**Advantages:**
- Model-agnostic (works with any classifier)
- Local explanations (specific to individual prediction)
- Intuitive (local linear approximation)

**Disadvantages:**
- Requires additional computation
- Results depend on random seed
- Different from global feature importance

**Configuration:**
- `num_features`: Number of features to return (default: 10)
- `num_permutations`: Number of variations to generate (LIME default)

### Method 2: Linear Weights (Fallback)

**Location:** `src/xai_explainer.py` → `_explain_with_linear_weights()`

**Algorithm:**
1. Vectorize email using TF-IDF
2. Extract model coefficients for each feature
3. For phishing class (label=1):
   - contribution = TF-IDF weight × model coefficient
4. Sort by absolute contribution
5. Return top K features (words)

**Advantages:**
- No additional computation (uses pre-trained model)
- Global and consistent
- Works even if LIME is unavailable
- Fast

**Disadvantages:**
- Global explanation (not specific to individual email)
- Assumes linear separability

**Flow:**
```
Email Text
    ↓
[TF-IDF Vectorization] → Dense vector (5000,)
    ↓
[Extract Coefficients] → Model coef_ shape (1, 5000)
    ↓
[Element-wise Product]  → contribution = tfidf_weight * coefficient
    ↓
[Sort by Magnitude]     → Top features by importance
    ↓
[Return Top K]          → [(term1, weight1), ..., (termK, weightK)]
```

### Method Selection Logic

```python
if use_lime and LIME_AVAILABLE:
    explanation = _explain_with_lime(text, num_features)
else:
    explanation = _explain_with_linear_weights(text, num_features)
```

**LIME is preferred** over linear weights because it provides local explanations specific to the email being classified.

---

## MITRE Mapping Strategy

### Current Implementation

**Location:** `src/attacker_sim.py` → `mitre_mapping()`

**Current Logic:**
- Technique candidates are scored using pattern matches and keyword matches.
- Default mode returns a single best match.
- Multi-label mode returns all matches sorted by score.
- Detail mode (`return_details=True`) returns:
    - primary technique
    - alternative techniques
    - per-technique scores
    - confidence value derived from score concentration

**Coverage:**
- **T1566.002 (Phishing: Link)**: URLs detected via http://, https://, www., "click here"
- **T1566.001 (Phishing: Attachment/Generic)**: Fallback for any non-URL phishing

### Enhanced Mapping (Implemented)

**Proposed Extension Pattern:**

```python
MITRE_PATTERNS = {
    "T1566.002": {
        "name": "Phishing: Link",
        "patterns": [r"http\://", r"https\://", r"www\.", "click here", "click link"],
        "keywords": ["verify", "confirm", "update", "urgent"]
    },
    "T1566.001": {
        "name": "Phishing: Attachment",
        "patterns": ["attached", "document", "file", "invoice", "receipt"],
        "keywords": ["download", "open", "view"]
    },
    "T1598.003": {
        "name": "Phishing: Spearphishing Link",
        "patterns": [r"linkedin\.", r"facebook\.", "social media"],
        "keywords": ["connect", "profile", "invite"]
    },
    "T1598.001": {
        "name": "Spearphishing Attachment",
        "patterns": [".pdf", ".doc", ".xls"],
        "keywords": ["reviewed", "signed", "approved"]
    },
    "T1598.002": {
        "name": "Spearphishing Link (specific)",
        "patterns": [r"shorten\.link", r"bit\.ly", r"tinyurl"],
        "keywords": ["shortened", "track"]
    }
}
```

**Multi-label detection:**
```python
def mitre_mapping_enhanced(email_text: str) -> List[str]:
    techniques = []
    for technique, rules in MITRE_PATTERNS.items():
        if any(pattern in email_text.lower() for pattern in rules['patterns']):
            techniques.append(technique)
    
    if not techniques:
        techniques.append("T1566.001")  # Fallback
    
    return techniques
```

**Benefits:**
- Better coverage of MITRE ATT&CK phishing taxonomy
- Multi-label mapping (one email can match multiple techniques)
- Pattern-based and keyword-based matching
- Documented mapping rationale

### MITRE Mapping Validation Architecture

**Location:** `src/evaluate_mitre_mapping.py`

Validation flow:

1. Load manually labeled validation subset (`data/processed/mitre_validation_subset.csv`)
2. Compute primary-label predictions and multi-label predictions
3. Report:
    - primary accuracy
    - macro precision/recall/F1
    - multi-label exact-match
    - micro precision/recall/F1
4. Persist outputs to `evaluation_results/mitre_mapping_predictions.csv` and `evaluation_results/mitre_mapping_summary.csv`

This creates an empirical quality checkpoint for the mapping subsystem.

---

## Quality Engineering Layer

### Reproducibility Controls

- One-command pipeline execution (`src/reproduce_pipeline.py`)
- Seeded execution and metadata capture (`simulation_results/reproducibility_run_metadata.json`)
- Pinned dependency versions in `requirements.txt`

### Runtime Compatibility Guard

Model loading paths now compare runtime scikit-learn version to model training version metadata and emit warnings when they differ.

### Continuous Integration

- Workflow: `.github/workflows/ci.yml`
- Action: install dependencies and run `pytest -q`
- Objective: prevent regression drift and keep the test baseline enforceable

---

## Adversarial Simulation

### Purpose

Test model robustness against real-world phishing attacks by:
- Applying mutation rules to legitimate phishing emails
- Generating single-rule and multi-rule attack variants
- Measuring detection rate and bypass rate
- Analyzing where the model fails

### Attack Rules

**Location:** `src/attacker_sim.py` → `ATTACK_RULES`

| Rule | Function | Purpose |
|------|----------|---------|
| `urgency` | Add deadline + urgency language | Exploit time pressure |
| `spoof_bank` | Inject bank name + authority | Impersonate financial institution |
| `spoof_revenue` | Inject tax authority message | Social engineering authority |
| `fake_link` | Append malicious URL | Drive user to phishing site |

### Attack Variant Generation

```
Base Phishing Email
    ↓
[Single-Rule Attacks] → 4 single-rule variants
    - urgency
    - spoof_bank
    - spoof_revenue
    - fake_link
    ↓
[Two-Rule Chains] → 4 × 3 = 12 combinations
    - urgency + spoof_bank
    - urgency + spoof_revenue
    - ...etc
    ↓
Total: 16 attack variants per base email
```

### Simulation Flow

```python
for threshold in [0.1, 0.2, ..., 0.95]:  # 10 thresholds
    for round in range(5):                 # 5 rounds per threshold
        
        # Test each base phishing email + attack variants
        for base_email in BASE_PHISHING_EMAILS:
            for variant in generate_attack_variants(base_email):
                modified_text = apply_rules(base_email, variant)
                pred, prob = classify_email(modified_text, threshold)
                
                if pred == 0:  # False negative (attacker wins)
                    explanation = explain_email(modified_text)
                    log_attack_success(explanation)
        
        # Test legitimate emails (should not trigger)
        for legit_email in BASE_LEGIT_EMAILS:
            pred, prob = classify_email(legit_email, threshold)
```

### Metrics Computed

**Per threshold per round:**
- TP, FP, TN, FN (from confusion marix)
- Detection Rate = TP / (TP + FN) [% of phishing detected]
- Bypass Rate = FN / (TP + FN) [% of phishing missed]
- Precision = TP / (TP + FP) [accuracy of positive predictions]
- Recall = TP / (TP + FN) [same as detection rate]
- F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
- Accuracy = (TP + TN) / N [overall correctness]

### Analysis Pipeline

```
attacker_simulation_log.csv (raw logs)
        ↓
[analyse_simulation_results.py]
        ↓
Group by: (threshold, attack_type, rule_chain, mitre_technique)
        ↓
Compute metrics per group
        ↓
Summary CSVs:
  - summary_by_threshold.csv
  - summary_by_attack_type.csv
  - summary_by_rule_chain.csv
  - summary_by_mitre.csv
        ↓
[visualise_results.py]
        ↓
Generate figures:
  - threshold_detection_rate.png
  - attack_type_detection_rate.png
  - rule_chain_detection_rate.png
  - mitre_detection_rate.png
```

---

## Module Dependencies

### Dependency Graph

```
app.py (Streamlit UI)
  ├── mvp_baseline.py (load_model)
  ├── xai_explainer.py (explain_email)
  │   └── mvp_baseline.py
  ├── attacker_sim.py (mitre_mapping)
  └── visualise_results.py (figures)

attacker_sim.py
  ├── mvp_baseline.py (load_model, classify_email)
  ├── xai_explainer.py (explain_email)
  └── analyse_simulation_results.py (metrics)

mvp_baseline.py
  ├── sklearn (TF-IDF, LogReg, metrics)
  ├── lime (LimeTextExplainer)
  ├── shap (LinearExplainer)
  ├── joblib (save/load)
  └── matplotlib (ROC curve)

build_dataset.py
  └── pandas (load/concat CSVs)

analyse_simulation_results.py
  └── pandas (groupby, metrics)

visualise_results.py
  └── matplotlib (plotting)

xai_explainer.py
  ├── mvp_baseline.py (load_model)
  └── lime (LimeTextExplainer)
```

### Import Strategy

- **Lazy loading** of model in xai_explainer (only when explain_email called)
- **Cached model** in app.py using @st.cache_resource
- **Joblib** for persistent model serialization

---

## Configuration & Constants

### Global Constants

**Location:** Top of each module

```python
# mvp_baseline.py
DATA_PATH = "data/processed/english_dataset.csv"
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
MODEL_DIR = "models"
VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"
MODEL_PATH = "models/logreg_model.joblib"

LABEL_MAP = {
    0: "legit",
    1: "phishing",
}

# TF-IDF Constants
TFIDF_MAX_FEATURES = 5000
TFIDF_STOP_WORDS = "english"

# Model Constants
LR_MAX_ITER = 1000
LR_CLASS_WEIGHT = "balanced"

# attacker_sim.py
SIM_OUTPUT_DIR = "simulation_results"
SIM_OUTPUT_PATH = "simulation_results/attacker_simulation_log.csv"

THRESHOLDS = [0.10, 0.15, 0.20, ..., 0.95]  # Step of 0.05
NUM_ROUNDS = 5
NUM_FEATURES_FOR_EXPLANATION = 10

# app.py
DEFAULT_THRESHOLD = 0.5
DEFAULT_NUM_FEATURES = 10
FIGURES_DIR = "simulation_results/figures"
```

### Feature Engineering

**TF-IDF Parameters:**
- Lowercase: ✓ (normalize case)
- Stop words: English (remove "the", "and", etc.)
- Max features: 5000 (balance between coverage and sparsity)
- Min/max document frequency: default (1 doc to 100%)
- N-gram: unigrams only (1-gram, no bigrams)

**Rationale:**
- Stop words: Reduce noise from function words
- Max features: Top 5000 most discriminative terms
- Unigrams: Bigrams add complexity without significant improvement for phishing

---

## Third-Party Integrations

### LIME Integration

**Purpose:** Explain individual predictions
**When used:** In attacker simulation and app (when attack succeeds)
**Fallback:** Linear weights if LIME computation fails

### MITRE ATT&CK Framework

**Purpose:** Map detected phishing to official attack taxonomy
**Current coverage:** T1566.001 and T1566.002
**Extensibility:** Pattern matching in `mitre_mapping()` can be enhanced

### Streamlit

**Purpose:** Interactive UI for model demonstrations
**Components:**
- Text area for email input
- Sliders for threshold and feature count
- Columns for side-by-side display
- Image display for simulation results

---

## Performance Considerations

### Model Training (mvp_baseline.py)

| Step | Complexity | Duration |
|------|-----------|----------|
| Load data | O(N) | ~5 sec |
| TF-IDF fit | O(N × V) | ~30 sec |
| LogReg train | O(N × F) | ~5 sec |
| Evaluation | O(N) | ~2 sec |

N = samples, V = vocabulary, F = features

### Inference (per email)

| Step | Complexity | Duration |
|------|-----------|----------|
| TF-IDF transform | O(F) | <1 ms |
| LogReg predict | O(F) | <1 ms |
| LIME explain | O(P × F) | ~500 ms |
| Linear weights | O(F) | <1 ms |

F = features, P = permutations

### Storage Requirements

| Component | Size |
|-----------|------|
| Vectorizer | ~2 MB |
| Model | ~100 KB |
| Training CSV | dataset-dependent |
| Simulation log | ~50 MB (100K records) |

---

## Error Handling

### Model Loading Failures

```python
try:
    vectorizer, clf = load_model()
except FileNotFoundError:
    raise FileNotFoundError(
        "Model files not found. Run mvp_baseline.py to train first."
    )
```

### Explanation Failures

```python
try:
    if use_lime:
        explanation = _explain_with_lime(text, num_features)
except Exception as e:
    # Fallback to linear weights
    explanation = _explain_with_linear_weights(text, num_features)
```

### Data Validation

```python
if not email_text.strip():
    raise ValueError("Email text cannot be empty.")

if not isinstance(num_features, int) or num_features < 1:
    raise ValueError("num_features must be a positive integer.")
```

---

## Future Architecture Improvements

1. **Modular MITRE Mapping**: Move patterns to `mitre_mappings.json`
2. **Utility Module**: Extract `classify_email()` to `src/utils.py`
3. **Configuration File**: Central `config.yaml` for all constants
4. **Logging**: Add structured logging with Python `logging` module
5. **Async Processing**: Parallelize simulation if running large-scale
6. **Caching Layer**: Redis/pickle for model caching during simulation
7. **API Layer**: REST API wrapper around core functions
8. **Database**: Store simulation results in SQLite instead of CSV

---

**Document Version**: 1.0  
**Last Updated**: March 2026
