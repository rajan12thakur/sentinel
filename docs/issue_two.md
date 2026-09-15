SENTINEL — Issue #2: Dataset Ingestion, Profiling & Initial ML Baselines

Issue

Issue #2 — Dataset ingestion, data profiling, preprocessing, feature engineering, leakage-safe splitting, and initial failure-risk classification baselines

1. Purpose

This document records the implementation completed after the Django project foundation. The goal was to build a reproducible ML data pipeline using NASA C-MAPSS FD001 and establish the first failure-risk classification baselines.

The existing project plan defines C-MAPSS ingestion, EDA, leakage investigation, leakage-safe splitting, feature engineering, and classical ML baselines as core ML work. fileciteturn4file0L291-L313

2. Scope

Completed

C-MAPSS FD001 ingestion

schema definition and validation

dataset profiling

missing-value analysis

engine trajectory analysis

constant-feature detection/removal

RUL target generation

failure-risk target generation

rolling mean/std features

cycle-to-cycle delta features

engine-aware train/validation split

engine leakage validation

ML-ready train/validation datasets

Dummy classification baseline

Logistic Regression baseline

Random Forest baseline

classification evaluation metrics

baseline comparison

Deferred

XGBoost

RUL regression

PyTorch

MLP

1D CNN

LSTM

Transformer

focal loss / weighted sampling

SHAP

MLflow

prediction API

database prediction models

dashboard ML integration

Docker and deployment

This follows the project principle that model complexity and infrastructure should be introduced when justified by an actual requirement or experiment. fileciteturn4file0L25-L27

3. Dataset

Dataset:

NASA C-MAPSS turbofan engine degradation/prognostics dataset

Initial subset:

FD001

Expected raw files:

train_FD001.txt
test_FD001.txt
RUL_FD001.txt

Repository location:

data/raw/cmapss/

Raw datasets are not committed to Git.

4. Dataset Schema

The loader assigns these 26 columns:

unit_id
cycle
op_setting_1
op_setting_2
op_setting_3
sensor_1
sensor_2
sensor_3
sensor_4
sensor_5
sensor_6
sensor_7
sensor_8
sensor_9
sensor_10
sensor_11
sensor_12
sensor_13
sensor_14
sensor_15
sensor_16
sensor_17
sensor_18
sensor_19
sensor_20
sensor_21

5. Implemented Data Loader

File:

ml/data/loader.py

Responsibilities:

verify the requested file exists

read whitespace-separated C-MAPSS data

assign explicit column names

return a pandas DataFrame

keep raw data unchanged

Pipeline:

train_FD001.txt
       ↓
load_cmapss_train()
       ↓
DataFrame

6. Implemented Dataset Validation

File:

ml/data/validator.py

Checks include:

dataframe is not empty

all expected columns exist

unit_id has no missing values

cycle has no missing values

cycle values are positive

engine units exist

sensor values contain no missing values

This makes invalid input fail explicitly rather than silently entering model training.

7. Actual Dataset Profiling Results

Experiment:

ml/experiments/02_explore_cmapss.py

Command:

python -m ml.experiments.02_explore_cmapss

Observed:

Rows: 20,631
Columns: 26
Engine units: 100
Cycle range: 1 - 362

Missing-value result:

No missing values found.

8. Engine Trajectory Analysis

Observed engine-length statistics:

count    100
mean     206.310000
std       46.342749
min      128
25%      177
50%      199
75%      229.25
max      362

The engines therefore have different trajectory lengths.

This is important because the split must respect engine identity rather than treating all rows as independent observations.

9. Constant Feature Analysis

The following raw sensors were constant:

sensor_1
sensor_5
sensor_10
sensor_16
sensor_18
sensor_19

op_setting_3 was also constant in FD001.

The feature-selection implementation detects constant features automatically instead of hardcoding this list.

File:

ml/features/selection.py

10. RUL Target

File:

ml/data/preprocessing.py

RUL is calculated independently for each engine:

RUL = maximum_cycle_for_engine - current_cycle

Resulting RUL range:

0 → 361

Overall dataset statistics:

count    20631.000000
mean       107.807862
std         68.880990
min          0.000000
25%         51.000000
50%        103.000000
75%        155.000000
max        361.000000

11. Failure-Risk Target

Initial three-class formulation:

RUL <= 30
    → high

30 < RUL <= 75
    → medium

RUL > 75
    → low

These are initial modeling/business assumptions and are not claimed to be universal industrial thresholds.

Initial distribution:

low       13,031
medium     4,500
high       3,100

12. Feature Engineering

File:

ml/features/engineering.py

Current features include:

Raw sensor values

sensor_X

Rolling statistics

Using a five-cycle window:

sensor_X_rolling_mean
sensor_X_rolling_std

Cycle-to-cycle deltas

sensor_X_delta

The rolling and delta calculations are performed independently per engine.

The feature-generation implementation was also corrected so delta features are generated only from original sensor columns, avoiding accidental recursive features such as deltas of rolling features.

13. Engine-Aware Train/Validation Split

File:

ml/data/splitting.py

Implementation:

GroupShuffleSplit

Configuration:

validation_size = 0.2
random_state = 42
group = unit_id

Result:

Training engines: 80
Validation engines: 20

Training rows: 16,561
Validation rows: 4,070

14. Leakage Prevention

A dedicated function:

validate_engine_split()

checks for overlap between training and validation engine IDs.

Actual result:

Engine leakage check: PASSED

This prevents the same engine's degradation trajectory from appearing in both datasets.

The project methodology explicitly requires checking whether the same machine is present in train and validation and whether future information enters historical features. fileciteturn4file0L375-L397

15. ML Dataset Preparation

Experiment:

ml/experiments/04_prepare_ml_dataset.py

Pipeline:

Raw C-MAPSS
     ↓
RUL generation
     ↓
Risk-label generation
     ↓
Engine-aware split
     ↓
Feature engineering
     ↓
Constant-feature removal
     ↓
Feature/target separation
     ↓
Processed train/validation datasets

Generated files:

data/processed/train.csv
data/processed/validation.csv

Processed datasets are generated artifacts and should remain ignored by Git.

16. Final Feature Count

After feature preparation and constant-feature removal:

Training rows: 16,561
Validation rows: 4,070

Training features: 63
Validation features: 63

Total removed constant features:

25

Removed features:

op_setting_3

sensor_1
sensor_5
sensor_10
sensor_16
sensor_18
sensor_19

sensor_1_rolling_mean
sensor_1_rolling_std
sensor_5_rolling_mean
sensor_5_rolling_std
sensor_10_rolling_mean
sensor_10_rolling_std
sensor_16_rolling_mean
sensor_16_rolling_std
sensor_18_rolling_mean
sensor_18_rolling_std
sensor_19_rolling_mean
sensor_19_rolling_std

sensor_1_delta
sensor_5_delta
sensor_10_delta
sensor_16_delta
sensor_18_delta
sensor_19_delta

17. Feature / Target Separation

File:

ml/data/dataset.py

Excluded from model features:

unit_id
rul
risk_level

unit_id is retained separately for traceability but is not supplied to the model.

Final conceptual structure:

X
├── operating features
├── raw sensor features
├── rolling features
└── delta features

y_risk
y_rul
engine_ids

18. Classification Baseline

Experiment:

ml/experiments/05_classification_baselines.py

Models implemented:

1. DummyClassifier
2. Logistic Regression
3. Random Forest

Metrics:

Accuracy
Macro Precision
Macro Recall
Macro F1
Weighted F1
Macro PR-AUC
Confusion Matrix
Classification Report

The project methodology explicitly states that accuracy should not be the only metric for an imbalanced failure-risk problem. fileciteturn4file0L315-L343

19. Dummy Baseline Results

Validation:

Accuracy       : 0.6265
Macro Precision: 0.2088
Macro Recall   : 0.3333
Macro F1       : 0.2568
Weighted F1    : 0.4827
Macro PR-AUC   : 0.3333

The model predicted every observation as:

low

Confusion matrix:

[[   0    0  620]
 [   0    0  900]
 [   0    0 2550]]

This demonstrates why a high accuracy number can be misleading when the majority class dominates.

20. Logistic Regression Results

Configuration:

StandardScaler
LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

Validation results:

Accuracy        : 0.8474
Macro Precision : 0.8036
Macro Recall    : 0.8431
Macro F1        : 0.8200
Weighted F1     : 0.8526
Macro PR-AUC    : 0.5354

Confusion matrix:

[[ 564   56    0]
 [ 119  679  102]
 [   0  344 2206]]

Per-class results:

high
precision: 0.83
recall:    0.91
f1:        0.87

medium
precision: 0.63
recall:    0.75
f1:        0.69

low
precision: 0.96
recall:    0.87
f1:        0.91

21. Random Forest Results

Configuration:

n_estimators=300
min_samples_leaf=2
class_weight="balanced"
random_state=42
n_jobs=-1

Validation results:

Accuracy        : 0.8725
Macro Precision : 0.8451
Macro Recall    : 0.8477
Macro F1        : 0.8461
Weighted F1     : 0.8739
Macro PR-AUC    : 0.5268

Confusion matrix:

[[ 548   72    0]
 [  57  670  173]
 [   0  217 2333]]

Per-class results:

high
precision: 0.91
recall:    0.88
f1:        0.89

medium
precision: 0.70
recall:    0.74
f1:        0.72

low
precision: 0.93
recall:    0.91
f1:        0.92

22. Baseline Comparison

Actual validation results:

Model                  Accuracy   Macro F1   Macro Recall   PR-AUC
------------------------------------------------------------------
Dummy                  0.6265     0.2568     0.3333         0.3333
Logistic Regression    0.8474     0.8200     0.8431         0.5354
Random Forest          0.8725     0.8461     0.8477         0.5268

Initial interpretation:

Both real ML models substantially outperform the dummy baseline.

Random Forest currently has the best accuracy and macro F1.

Logistic Regression currently has slightly higher macro PR-AUC.

High-risk recall is strong for both models.

Medium-risk classification is more difficult than high/low risk.

These are baseline results only.

No final model selection should be made from these results alone.

23. Important Modeling Principle

The project will not optimize for a visually impressive metric.

The intended workflow is:

Train
  ↓
Evaluate
  ↓
Compare
  ↓
Analyze errors
  ↓
Improve
  ↓
Select based on evidence

If a simpler model performs better than a deep model, the simpler model can legitimately remain the selected model. This is consistent with the project's stated evaluation philosophy. fileciteturn4file0L509-L537

24. Current Repository Structure

Relevant structure:

sentinel/
│
├── data/
│   ├── raw/
│   │   └── cmapss/
│   │       ├── train_FD001.txt
│   │       ├── test_FD001.txt
│   │       └── RUL_FD001.txt
│   │
│   ├── processed/
│   │   ├── train.csv
│   │   └── validation.csv
│   │
│   └── external/
│
├── ml/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── loader.py
│   │   ├── preprocessing.py
│   │   ├── splitting.py
│   │   └── validator.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engineering.py
│   │   └── selection.py
│   │
│   ├── models/
│   │   └── __init__.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── classification.py
│   │
│   ├── experiments/
│   │   ├── .gitkeep
│   │   ├── 01_prepare_cmapss.py
│   │   ├── 02_explore_cmapss.py
│   │   ├── 03_create_split.py
│   │   ├── 04_prepare_ml_dataset.py
│   │   └── 05_classification_baselines.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── constants.py
│
├── core/
├── dashboard/
├── machines/
├── predictions/
├── ml_models/
├── templates/
├── static/
│
├── manage.py
├── requirements.txt
├── .env.example
├── .gitignore
└── docker-compose.yml

25. Git Rules for This Issue

Commit:

Python source
experiment scripts
evaluation code
documentation
configuration changes

Do not commit:

.env
.venv/
data/raw/
data/external/
data/processed/
mlruns/
generated experiment artifacts

The repository workflow remains:

Issue branch
     ↓
GitHub branch
     ↓
Pull Request
     ↓
develop
     ↓
release PR
     ↓
main

Never send an issue branch directly to main. fileciteturn4file0L1153-L1215

26. Validation Commands

Run before the PR:

python -m ml.experiments.01_prepare_cmapss

python -m ml.experiments.02_explore_cmapss

python -m ml.experiments.03_create_split

python -m ml.experiments.04_prepare_ml_dataset

python -m ml.experiments.05_classification_baselines

Django checks:

python manage.py check
python manage.py test

27. Definition of Done

C-MAPSS FD001 obtained

Raw data placed under data/raw/cmapss/

Data loader implemented

Schema validation implemented

Dataset profiling completed

Missing-value analysis completed

Engine trajectory analysis completed

Constant-feature analysis completed

RUL generation implemented

Risk-label generation implemented

Rolling features implemented

Delta features implemented

Engine-aware split implemented

Engine leakage validation implemented

Constant-feature removal implemented

ML-ready datasets generated

Dummy classification baseline implemented

Logistic Regression baseline implemented

Random Forest baseline implemented

Classification evaluation implemented

Baseline results recorded

XGBoost baseline

RUL regression baseline

PyTorch training pipeline

Temporal deep learning

Explainability

MLflow

Prediction API

28. Known Limitations

Only FD001 is currently used.

Risk thresholds are initial assumptions.

XGBoost has not yet been evaluated.

RUL regression has not yet been evaluated.

No independent final test evaluation has been performed.

No hyperparameter optimization has been performed.

No calibration analysis has been performed.

No explainability analysis has been performed.

No model artifact/versioning system has been implemented yet.

Processed datasets are generated locally.

Deep sequence models have not yet been implemented.

29. Next Development Step

The next issue should continue from the established baseline rather than jumping immediately to deep learning.

Recommended next work:

Classification baseline
        ↓
XGBoost comparison
        ↓
RUL regression baseline
        ↓
MAE / RMSE / R²
        ↓
Error analysis
        ↓
Then PyTorch

The original project roadmap separates classical failure-risk modeling from RUL regression and later PyTorch/temporal modeling. fileciteturn4file0L1407-L1459

30. Engineering Summary

At the end of this issue, SENTINEL has moved from a Django/API foundation to a reproducible ML data pipeline.

Current flow:

C-MAPSS FD001
      ↓
Data Loader
      ↓
Validation
      ↓
EDA
      ↓
RUL Generation
      ↓
Risk Labels
      ↓
Engine-Aware Split
      ↓
Feature Engineering
      ↓
Constant Feature Removal
      ↓
Prepared ML Dataset
      ↓
Dummy Baseline
      ↓
Logistic Regression
      ↓
Random Forest

The project now has actual experimental evidence showing that the engineered sensor features contain useful signal for failure-risk classification.

The next goal is to extend the baseline scientifically, not simply add more technologies.