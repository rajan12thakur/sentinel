SENTINEL — Issue #11

Build Remaining Useful Life (RUL) Regression Baseline

1. Issue Overview

Issue: #11 — Build Remaining Useful Life (RUL) Regression Baseline

Branch: 11-build-remaining-useful-life-rul-regression-baseline

Objective: Build and evaluate classical machine-learning regression models for predicting the Remaining Useful Life (RUL) of industrial engines using the prepared NASA C-MAPSS FD001 dataset.

This issue establishes a reliable classical baseline before moving to custom PyTorch and temporal deep-learning approaches.

The objective is to establish:

a reproducible RUL regression dataset,

classical regression baselines,

predictive-maintenance-focused evaluation,

low-RUL performance analysis,

engine-level robustness analysis,

model comparison,

feature-importance analysis,

diagnostic plots,

and a benchmark for future deep-learning experiments.

2. Why RUL Regression?

Predictive maintenance requires estimating how much useful operating life remains before a machine reaches failure.

SENTINEL contains two complementary prediction tasks.

Risk classification

Answers:

How risky is the machine right now?

Example:

Low Risk
Medium Risk
High Risk

RUL regression

Answers:

Approximately how many operating cycles remain?

Example:

Predicted RUL = 18 cycles

The two outputs therefore provide different maintenance information.

Conceptually:

Sensor Data
     |
     +--------------------+
     |                    |
     v                    v
Risk Classification    RUL Regression
     |                    |
     v                    v
Risk Level             Remaining Life

3. Dataset

The experiment uses the prepared NASA C-MAPSS FD001 dataset.

The raw FD001 training dataset contains:

20,631 rows

100 engine units

26 raw columns

operating settings

multiple sensor measurements

engine identifiers

operating cycles

The processed datasets used by this issue are:

data/processed/train.csv
data/processed/validation.csv

These generated datasets are not intended to be committed directly to Git.

4. Dataset Preparation

The processed dataset contains:

engine identifier,

engineered sensor features,

RUL target,

risk-level target.

For RUL regression, the following columns are excluded from the model input:

unit_id
rul
risk_level

The remaining columns form the regression feature matrix.

This prevents the model from directly receiving the target or engine identifier.

After feature selection:

Training features:    63
Validation features:  63

A total of:

25 constant features

were removed during the previous feature-preparation stage.

5. Train/Validation Strategy

The validation dataset was created using an engine-level group split.

The key rule is:

Rows from the same engine must not appear in both training and validation datasets.

This prevents engine-level leakage.

The split produced:

Training rows:       16,561
Validation rows:      4,070

Training engines:         80
Validation engines:       20

Therefore:

80 engines → training
20 engines → validation

A random row split would be inappropriate because observations from the same engine are strongly related over time. It could allow the model to see an engine during training and validation, resulting in overly optimistic validation performance.

6. RUL Target Distribution

Training RUL statistics:

Count: 16,561
Mean:  108.3778
Std:    69.6399
Min:     0
Max:   361

Validation RUL statistics:

Count: 4,070
Mean:  105.4887
Std:    65.6600
Min:     0
Max:   268

The validation set does not need to have the same maximum RUL as the training set because it contains a separate group of engines.

7. Models Evaluated

Four regression approaches were evaluated.

7.1 Dummy Regressor

The first baseline uses the training-set mean:

DummyRegressor(
    strategy="mean"
)

This model does not learn relationships between sensor features and RUL.

Its purpose is to answer:

Do the actual machine-learning models learn useful information compared with simply predicting the average RUL?

7.2 Ridge Regression

The second baseline is Ridge regression.

Pipeline:

Features
   |
   v
StandardScaler
   |
   v
Ridge Regression

Configuration:

Ridge(
    alpha=1.0
)

Standardization is used because Ridge regression is sensitive to feature scale.

Ridge provides a simple linear baseline and shows how much predictive information can be captured without nonlinear models.

7.3 Random Forest Regression

The third model is Random Forest regression.

Configuration:

RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

Important choices:

n_estimators=300 provides a reasonably stable ensemble.

max_depth=None allows trees to grow subject to the other stopping conditions.

min_samples_leaf=2 provides mild regularization.

random_state=42 makes the experiment reproducible.

n_jobs=-1 enables parallel CPU execution.

Random Forest can model nonlinear relationships without requiring feature scaling.

7.4 XGBoost Regression

The fourth model is XGBoost.

Configuration:

XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)

This provides a strong gradient-boosted-tree comparison against Random Forest.

8. Evaluation Metrics

The experiment uses multiple regression metrics.

8.1 MAE

Mean Absolute Error:

MAE = mean(|actual - predicted|)

MAE answers:

On average, how many cycles away is the prediction?

Lower is better.

8.2 RMSE

Root Mean Squared Error:

RMSE = sqrt(mean((actual - predicted)^2))

RMSE penalizes larger errors more heavily.

Lower is better.

8.3 R²

R² measures the amount of target variance explained by the model relative to a mean baseline.

R² = 1.0  → perfect
R² = 0.0  → mean-baseline level
R² < 0.0  → worse than the mean baseline

Higher is better.

8.4 Median Absolute Error

This measures the median absolute prediction error.

It is less affected by extreme outliers than MAE.

Lower is better.

8.5 Maximum Absolute Error

This records the largest absolute prediction error.

It is useful for understanding worst-case behavior, although it is not used as the sole model-selection metric.

9. Overall Regression Results

Actual validation results:

Model

MAE

RMSE

R²

Median Absolute Error

Maximum Absolute Error

Dummy

55.3633

65.7154

-0.0019

51.6222

159.6222

Ridge

25.1800

31.6938

0.7669

21.4612

122.4318

Random Forest

23.4629

31.7665

0.7659

17.0754

108.2515

XGBoost

23.6209

31.6505

0.7676

17.4725

105.5763

10. Overall Results Interpretation

Dummy

MAE = 55.3633
R²  = -0.0019

The dummy model provides a weak baseline and confirms that the learned models are extracting useful information from the features.

Ridge

MAE = 25.1800
RMSE = 31.6938
R² = 0.7669

Ridge performs substantially better than the dummy baseline.

This demonstrates that the engineered features contain strong predictive information even for a regularized linear model.

Random Forest

MAE = 23.4629
RMSE = 31.7665
R² = 0.7659
Median AE = 17.0754
Max AE = 108.2515

Random Forest achieves the best overall MAE and median absolute error among the evaluated models.

XGBoost

MAE = 23.6209
RMSE = 31.6505
R² = 0.7676
Median AE = 17.4725
Max AE = 105.5763

XGBoost has slightly better RMSE, R², and maximum error than Random Forest.

However, its overall MAE is slightly worse.

11. Classical Baseline Selection

For the current SENTINEL pipeline, Random Forest is selected as the primary classical RUL baseline.

The decision is based primarily on:

best overall MAE,

best median absolute error,

strong low-RUL performance,

slightly better mean engine-level MAE.

This does not mean Random Forest wins every metric.

XGBoost performs better on:

RMSE,

R²,

maximum absolute error,

worst-engine MAE.

Therefore the defensible conclusion is:

Random Forest is selected as the primary classical RUL baseline because it provides the best overall MAE and median error, slightly better low-RUL performance, and slightly better average engine-level performance. XGBoost remains a strong alternative with marginal advantages on several worst-case-oriented metrics.

12. Low-RUL Analysis

Overall performance is not sufficient for predictive maintenance.

A model can have acceptable average performance while performing poorly when a machine is close to failure.

Therefore, a dedicated low-RUL evaluation is performed for:

RUL <= 30 cycles

There were:

620 low-RUL validation samples

13. Low-RUL Results

Model

Samples

MAE

RMSE

Dummy

620

93.3778

93.8052

Ridge

620

20.1978

24.3749

Random Forest

620

4.6743

7.1386

XGBoost

620

4.9552

7.3001

14. Low-RUL Interpretation

Random Forest achieves:

MAE = 4.6743 cycles

in the low-RUL region.

This is considerably better than:

Ridge MAE = 20.1978
Dummy MAE = 93.3778

Random Forest also slightly outperforms XGBoost:

Random Forest = 4.6743
XGBoost       = 4.9552

This is particularly important because accurate estimation near failure is highly relevant to maintenance decisions.

15. Why Low-RUL Performance Matters

The same numerical error can have different operational implications depending on the machine lifecycle stage.

For example:

Actual RUL = 250
Predicted  = 245
Error      = 5 cycles

is generally less urgent than:

Actual RUL = 8
Predicted  = 30
Error      = 22 cycles

The second error could delay maintenance action.

Therefore, SENTINEL evaluates low-RUL performance separately rather than relying only on global MAE.

16. Engine-Level Evaluation

Row-level metrics can hide poor performance on individual engines.

For this reason, validation predictions are grouped by:

unit_id

For every engine, the experiment calculates:

Engine MAE
Number of validation samples

This determines whether model performance is consistent across different engine trajectories.

17. Random Forest Engine-Level Results

Random Forest:

Mean Engine MAE   = 23.9262
Median Engine MAE = 22.5748
Best Engine MAE   = 5.9977
Worst Engine MAE  = 52.4097

Per-engine results:

Engine

MAE

Samples

1

24.7187

192

5

19.2215

269

11

12.7699

240

13

21.5046

163

19

22.3653

158

23

23.5552

168

31

23.9777

234

32

9.2222

191

34

6.4923

195

40

9.1619

188

45

52.4097

158

46

31.7084

256

54

22.7842

257

71

5.9977

208

74

34.6879

166

77

34.8584

154

78

17.8783

231

81

21.6716

240

84

44.8440

267

91

38.6938

135

18. XGBoost Engine-Level Results

XGBoost:

Mean Engine MAE   = 24.0402
Median Engine MAE = 23.2208
Best Engine MAE   = 7.0656
Worst Engine MAE  = 45.9720

Per-engine results:

Engine

MAE

Samples

1

27.2781

192

5

19.0129

269

11

13.6298

240

13

19.3993

163

19

20.7666

158

23

24.1928

168

31

25.7512

234

32

10.1150

191

34

7.0656

195

40

10.6832

188

45

45.9720

158

46

27.6398

256

54

25.5321

257

71

7.1003

208

74

33.5289

166

77

34.0574

154

78

19.7696

231

81

22.2487

240

84

44.0472

267

91

43.0140

135

19. Engine-Level Interpretation

Performance is not uniform across engines.

Random Forest performs particularly well on some engines:

Engine 71 → MAE 5.9977
Engine 34 → MAE 6.4923
Engine 40 → MAE 9.1619

Some engines are substantially harder:

Engine 45 → MAE 52.4097
Engine 84 → MAE 44.8440
Engine 91 → MAE 38.6938

This indicates that individual degradation trajectories can have different levels of difficulty.

This analysis is valuable because an aggregate validation score would hide this variation.

20. Random Forest vs XGBoost at Engine Level

Mean engine MAE:

Random Forest = 23.9262
XGBoost       = 24.0402

Random Forest is slightly better on average.

However, XGBoost has a better worst-engine MAE:

Random Forest = 52.4097
XGBoost       = 45.9720

Therefore:

Random Forest has slightly better average engine-level performance.

XGBoost has better worst-engine performance.

The difference is not large enough to replace the primary baseline based on this analysis alone.

21. Prediction Diagnostics

Three diagnostic plots are generated for Random Forest.

21.1 Actual vs Predicted RUL

Axes:

X-axis → Actual RUL
Y-axis → Predicted RUL

A perfect model would place predictions along:

Predicted RUL = Actual RUL

The plot is useful for identifying:

systematic underprediction,

systematic overprediction,

prediction compression,

outliers,

behavior at different RUL ranges.

The observed predictions show strong agreement in the lower-RUL region, with more compression at higher RUL values.

21.2 Prediction Error Distribution

The second plot uses:

Prediction Error = Predicted RUL - Actual RUL

It helps identify:

whether errors are centered around zero,

systematic bias,

large outliers,

the overall spread of prediction errors.

21.3 Absolute Error vs Actual RUL

This plot compares:

X-axis → Actual RUL
Y-axis → Absolute Error

It helps determine whether model error changes with machine lifecycle stage.

The analysis indicates that error becomes more variable at higher RUL values, while the low-RUL region is predicted much more accurately by the tree-based models.

22. Random Forest Feature Importance

The final analysis added to this issue extracts feature importance from the trained Random Forest model.

The implementation uses:

model.feature_importances_

and produces a ranked table containing the top 15 features.

Conceptually:

Feature
   ↓
Random Forest importance
   ↓
Sort descending
   ↓
Top 15 features

This helps answer:

Which engineered sensor features were most useful for RUL prediction?

The analysis is useful for model interpretation and for understanding which engineered signals the classical model relies on.

23. Feature Importance Limitation

Random Forest feature importance should not be interpreted as causal evidence.

A high importance means that the feature was useful to the trained model.

It does not prove:

Feature X causes engine degradation.

Also, correlated features can share or redistribute importance.

More advanced explainability methods such as SHAP are planned for a later stage.

24. Reproducibility

Fixed random seeds are used where applicable.

Random Forest:

random_state=42

XGBoost:

random_state=42

The engine-level split was also generated using a fixed random state.

This allows the experiment to be reproduced consistently when using the same data and environment.

25. Code Structure

Main experiment:

ml/experiments/06_rul_regression_baselines.py

Reusable regression evaluation:

ml/evaluation/regression.py

Input datasets:

data/processed/train.csv
data/processed/validation.csv

The experiment flow is:

Load datasets
      ↓
Separate features and target
      ↓
Train Dummy
      ↓
Train Ridge
      ↓
Train Random Forest
      ↓
Train XGBoost
      ↓
Overall evaluation
      ↓
Low-RUL evaluation
      ↓
Engine-level evaluation
      ↓
Feature importance
      ↓
Diagnostic plots

26. Evaluation Module

The reusable regression evaluator calculates:

MAE
RMSE
R²
Median Absolute Error
Maximum Absolute Error
Predictions

This provides a consistent evaluation interface for different regression models.

Future PyTorch models can use the same evaluation philosophy so that comparisons remain meaningful.

27. Why the Validation Set Is Not Used for Training

The validation dataset is used only for evaluation.

The current issue does not perform extensive hyperparameter optimization against the validation set.

This helps preserve the validation engines as a meaningful holdout.

Repeatedly tuning against the same validation set could gradually overfit model decisions to that holdout.

28. What Was Not Done

The following are intentionally outside Issue #11:

extensive hyperparameter optimization,

neural networks,

custom PyTorch training loops,

LSTM,

Transformer,

multi-task learning,

SHAP,

MLflow,

model serving,

PostgreSQL prediction persistence,

FastAPI/Django inference endpoints,

Docker production deployment.

These belong to later stages of the project.

29. Why We Did Not Jump Directly to Deep Learning

SENTINEL is intended to demonstrate deeper ML engineering capabilities, but a complex model should not be introduced without a reliable baseline.

The progression is:

Dummy
  ↓
Linear model
  ↓
Tree-based models
  ↓
Custom PyTorch MLP
  ↓
1D CNN
  ↓
LSTM / temporal model
  ↓
Optional Transformer

This makes future improvements measurable.

For example, if a future LSTM achieves:

MAE = 18

it can be compared directly against:

Random Forest MAE = 23.4629

The model complexity is therefore justified only if it improves the required evaluation criteria.

30. Current Classical Benchmark

The primary classical benchmark is:

Random Forest Regressor

Overall:

MAE  = 23.4629
RMSE = 31.7665
R²   = 0.7659

Low-RUL:

MAE  = 4.6743
RMSE = 7.1386

Mean engine-level MAE:

23.9262

These values become the reference point for future RUL models.

31. Future Multi-Task Direction

SENTINEL eventually aims to combine:

RUL Regression
+
Risk Classification

using a shared representation.

Conceptually:

Sensor / Engine Features
          |
          v
   Shared ML Encoder
       /             /              v           v
RUL Head      Risk Head
     |           |
     v           v
RUL Value    Risk Level

This work is intentionally deferred until the individual regression and classification baselines are established.

32. Engineering Decisions

Decision 1 — Group-Based Validation

Reason: Prevent engine-level leakage.

Decision 2 — Dedicated Low-RUL Evaluation

Reason: Performance near failure is especially relevant to predictive maintenance.

Decision 3 — Engine-Level Evaluation

Reason: Aggregate metrics can hide poorly performing engine trajectories.

Decision 4 — Multiple Classical Models

Reason: Establish a meaningful benchmark before deep learning.

Decision 5 — Separate Validation Set

Reason: Reduce the risk of fitting model decisions to the holdout.

Decision 6 — Reproducible Random Seeds

Reason: Make experiments repeatable.

Decision 7 — Feature Importance

Reason: Understand which engineered features contribute most to the classical model.

33. Files Involved

ml/experiments/06_rul_regression_baselines.py
ml/evaluation/regression.py

Generated datasets:

data/processed/train.csv
data/processed/validation.csv

Generated datasets and experimental artifacts should remain Git-ignored where appropriate.

34. How to Run

From the SENTINEL project root:

python ml/experiments/06_rul_regression_baselines.py

The script will:

Load the processed datasets.

Print training and validation dimensions.

Train the Dummy Regressor.

Train Ridge Regression.

Train Random Forest.

Train XGBoost.

Print overall metrics.

Run low-RUL analysis.

Run engine-level analysis.

Print Random Forest feature importance.

Generate Random Forest diagnostic plots.

35. Definition of Done

Prepared RUL regression dataset loaded.

Target and identifier columns correctly separated.

Existing engine-level validation split reused.

Dummy regression baseline implemented.

Ridge regression implemented.

Random Forest regression implemented.

XGBoost regression implemented.

MAE implemented.

RMSE implemented.

R² implemented.

Median absolute error implemented.

Maximum absolute error implemented.

Overall model comparison completed.

Low-RUL analysis completed.

Engine-level evaluation completed.

Prediction diagnostics generated.

Random Forest feature importance added.

Reproducible configuration used.

Classical RUL baseline selected.

36. Final Conclusion

Issue #11 establishes the first complete classical RUL regression baseline for SENTINEL.

The experiment demonstrates that the engineered sensor features contain substantial predictive information about remaining useful life.

The Dummy model performs poorly, while Ridge, Random Forest, and XGBoost provide significant improvements.

The current primary classical baseline is Random Forest, with:

Overall MAE       = 23.4629
Overall RMSE      = 31.7665
Overall R²        = 0.7659
Median Error      = 17.0754
Maximum Error     = 108.2515

For the operationally important low-RUL region:

Low-RUL MAE       = 4.6743
Low-RUL RMSE      = 7.1386

The engine-level analysis shows that performance varies between engine trajectories, providing useful evidence for future robustness and temporal modeling.

XGBoost remains a strong alternative and performs slightly better on RMSE, R², maximum error, and worst-engine MAE. However, Random Forest has the best overall MAE, median absolute error, low-RUL MAE, and slightly better mean engine-level MAE.

The key outcome is therefore not simply one model score.

The issue establishes a:

reproducible, leakage-aware, predictive-maintenance-oriented RUL evaluation framework

that future PyTorch and temporal models can be compared against.

37. Next Step

The next major ML phase is:

Custom PyTorch RUL Model Training

The first deep-learning baseline should remain intentionally simple:

Prepared Features
       ↓
PyTorch Dataset
       ↓
DataLoader
       ↓
MLP
       ↓
RUL Prediction
       ↓
Training Loop
       ↓
Validation Loop
       ↓
Same Evaluation Metrics

After the MLP baseline:

MLP
 ↓
1D CNN
 ↓
LSTM / Temporal Model
 ↓
Optional Transformer Encoder

The Random Forest result from Issue #11 will be the benchmark that future models must beat to justify additional complexity.

