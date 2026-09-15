from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

from ml.evaluation.regression import (
    evaluate_regressor,
    print_regression_results,
)


# =====================================================
# Project Paths
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

TRAIN_FILE = (
    PROCESSED_DATA_DIR / "train.csv"
)

VALIDATION_FILE = (
    PROCESSED_DATA_DIR / "validation.csv"
)


# =====================================================
# Configuration
# =====================================================

TARGET_COLUMN = "rul"
ID_COLUMN = "unit_id"
RISK_COLUMN = "risk_level"

LOW_RUL_THRESHOLD = 30


# =====================================================
# Dataset Loading
# =====================================================

def load_datasets():
    """
    Load prepared training and validation datasets
    for RUL regression.
    """

    train_df = pd.read_csv(TRAIN_FILE)

    validation_df = pd.read_csv(VALIDATION_FILE)

    # Exclude identifiers and target columns
    # from the model features.
    feature_columns = [
        column
        for column in train_df.columns
        if column not in {
            ID_COLUMN,
            TARGET_COLUMN,
            RISK_COLUMN,
        }
    ]

    X_train = train_df[
        feature_columns
    ]

    y_train = train_df[
        TARGET_COLUMN
    ]

    X_validation = validation_df[
        feature_columns
    ]

    y_validation = validation_df[
        TARGET_COLUMN
    ]

    # Keep engine IDs for engine-level
    # error analysis.
    engine_ids = validation_df[
        ID_COLUMN
    ]

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        engine_ids,
    )


# =====================================================
# Low-RUL Evaluation
# =====================================================

def evaluate_low_rul_performance(
    y_true,
    predictions,
    threshold=LOW_RUL_THRESHOLD,
):
    """
    Evaluate model performance for samples
    with low remaining useful life.

    Low RUL is defined as RUL <= threshold.
    """

    low_rul_mask = y_true <= threshold

    y_true_low = y_true[
        low_rul_mask
    ]

    predictions_low = predictions[
        low_rul_mask
    ]

    if len(y_true_low) == 0:
        return None

    mae = mean_absolute_error(
        y_true_low,
        predictions_low,
    )

    rmse = (
        mean_squared_error(
            y_true_low,
            predictions_low,
        )
        ** 0.5
    )

    return {
        "samples": len(y_true_low),
        "mae": mae,
        "rmse": rmse,
    }


# =====================================================
# Engine-Level Evaluation
# =====================================================

def evaluate_engine_level_performance(
    engine_ids,
    y_true,
    predictions,
):
    """
    Evaluate RUL prediction error separately
    for each validation engine.

    Returns:
        engine_metrics:
            Per-engine MAE and sample count.

        summary:
            Mean, median, best, and worst
            engine-level MAE.
    """

    evaluation_df = pd.DataFrame(
        {
            "unit_id": engine_ids.values,
            "actual_rul": y_true.values,
            "predicted_rul": predictions,
        }
    )

    # Calculate absolute prediction error.
    evaluation_df["absolute_error"] = (
        evaluation_df["actual_rul"]
        - evaluation_df["predicted_rul"]
    ).abs()

    # Calculate MAE independently
    # for every engine.
    engine_metrics = (
        evaluation_df
        .groupby("unit_id")
        .agg(
            mae=(
                "absolute_error",
                "mean",
            ),
            samples=(
                "absolute_error",
                "count",
            ),
        )
        .reset_index()
    )

    # Summary across validation engines.
    summary = {
        "mean_engine_mae": engine_metrics[
            "mae"
        ].mean(),

        "median_engine_mae": engine_metrics[
            "mae"
        ].median(),

        "best_engine_mae": engine_metrics[
            "mae"
        ].min(),

        "worst_engine_mae": engine_metrics[
            "mae"
        ].max(),
    }

    return (
        engine_metrics,
        summary,
    )


# =====================================================
# Feature Importance Analysis
# =====================================================

def analyze_feature_importance(
    model,
    feature_names,
    top_n=15,
):
    """
    Display the most important features according
    to the trained Random Forest model.
    """

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    print("\nTop Feature Importances:")
    print(
        importance_df.to_string(
            index=False
        )
    )

    return importance_df


# =====================================================
# Prediction Analysis
# =====================================================

def analyze_predictions(
    model_name,
    y_true,
    predictions,
):
    """
    Generate regression diagnostic plots
    for a model's validation predictions.

    Plots:
    1. Actual vs Predicted RUL
    2. Prediction Error Distribution
    3. Absolute Error vs Actual RUL
    """

    errors = predictions - y_true

    absolute_errors = abs(
        errors
    )

    # -------------------------------------------------
    # 1. Actual vs Predicted RUL
    # -------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        y_true,
        predictions,
        alpha=0.4,
    )

    minimum = min(
        y_true.min(),
        predictions.min(),
    )

    maximum = max(
        y_true.max(),
        predictions.max(),
    )

    # Perfect prediction reference line.
    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
    )

    plt.xlabel(
        "Actual RUL"
    )

    plt.ylabel(
        "Predicted RUL"
    )

    plt.title(
        f"{model_name}: Actual vs Predicted RUL"
    )

    plt.tight_layout()

    plt.show()

    # -------------------------------------------------
    # 2. Prediction Error Distribution
    # -------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.hist(
        errors,
        bins=40,
    )

    plt.xlabel(
        "Prediction Error"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        f"{model_name}: Prediction Error Distribution"
    )

    plt.tight_layout()

    plt.show()

    # -------------------------------------------------
    # 3. Absolute Error vs Actual RUL
    # -------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        y_true,
        absolute_errors,
        alpha=0.4,
    )

    plt.xlabel(
        "Actual RUL"
    )

    plt.ylabel(
        "Absolute Error"
    )

    plt.title(
        f"{model_name}: Absolute Error vs Actual RUL"
    )

    plt.tight_layout()

    plt.show()


# =====================================================
# Main Experiment
# =====================================================

def main():

    print(
        "Loading prepared datasets..."
    )

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        validation_engine_ids,
    ) = load_datasets()

    print(
        f"Training shape: "
        f"{X_train.shape}"
    )

    print(
        f"Validation shape: "
        f"{X_validation.shape}"
    )

    print(
        f"Training RUL range: "
        f"{y_train.min()} - {y_train.max()}"
    )

    print(
        f"Validation RUL range: "
        f"{y_validation.min()} - {y_validation.max()}"
    )

    print(
        f"Validation engines: "
        f"{validation_engine_ids.nunique()}"
    )

    results = {}

    # =================================================
    # 1. Dummy Baseline
    # =================================================

    print(
        "\nTraining Dummy Regressor..."
    )

    dummy_model = DummyRegressor(
        strategy="mean"
    )

    dummy_model.fit(
        X_train,
        y_train,
    )

    dummy_results = evaluate_regressor(
        dummy_model,
        X_validation,
        y_validation,
    )

    results["Dummy"] = dummy_results

    print_regression_results(
        "Dummy Regressor",
        dummy_results,
    )

    # =================================================
    # 2. Ridge Regression
    # =================================================

    print(
        "\nTraining Ridge Regression..."
    )

    ridge_model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "regressor",
                Ridge(
                    alpha=1.0,
                ),
            ),
        ]
    )

    ridge_model.fit(
        X_train,
        y_train,
    )

    ridge_results = evaluate_regressor(
        ridge_model,
        X_validation,
        y_validation,
    )

    results["Ridge"] = ridge_results

    print_regression_results(
        "Ridge Regression",
        ridge_results,
    )

    # =================================================
    # 3. Random Forest
    # =================================================

    print(
        "\nTraining Random Forest..."
    )

    random_forest_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    random_forest_model.fit(
        X_train,
        y_train,
    )

    random_forest_results = evaluate_regressor(
        random_forest_model,
        X_validation,
        y_validation,
    )

    results["Random Forest"] = (
        random_forest_results
    )

    print_regression_results(
        "Random Forest",
        random_forest_results,
    )

    # =================================================
    # 4. XGBoost
    # =================================================

    print(
        "\nTraining XGBoost..."
    )

    xgboost_model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    xgboost_model.fit(
        X_train,
        y_train,
    )

    xgboost_results = evaluate_regressor(
        xgboost_model,
        X_validation,
        y_validation,
    )

    results["XGBoost"] = (
        xgboost_results
    )

    print_regression_results(
        "XGBoost",
        xgboost_results,
    )

    # =================================================
    # 5. Model Comparison
    # =================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RUL REGRESSION MODEL COMPARISON"
    )

    print(
        "=" * 70
    )

    summary_rows = []

    for model_name, result in results.items():

        summary_rows.append(
            {
                "model": model_name,
                "mae": result["mae"],
                "rmse": result["rmse"],
                "r2": result["r2"],
                "median_absolute_error": (
                    result[
                        "median_absolute_error"
                    ]
                ),
                "maximum_absolute_error": (
                    result[
                        "maximum_absolute_error"
                    ]
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    print(
        summary.to_string(
            index=False
        )
    )

    # =================================================
    # 6. Low-RUL Error Analysis
    # =================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LOW-RUL ERROR ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        f"Low-RUL threshold: "
        f"<= {LOW_RUL_THRESHOLD} cycles"
    )

    low_rul_summary_rows = []

    for model_name, result in results.items():

        low_rul_results = (
            evaluate_low_rul_performance(
                y_validation,
                result["predictions"],
                threshold=LOW_RUL_THRESHOLD,
            )
        )

        if low_rul_results is None:
            continue

        print(
            f"\n{model_name}"
        )

        print(
            f"Samples : "
            f"{low_rul_results['samples']}"
        )

        print(
            f"MAE     : "
            f"{low_rul_results['mae']:.4f}"
        )

        print(
            f"RMSE    : "
            f"{low_rul_results['rmse']:.4f}"
        )

        low_rul_summary_rows.append(
            {
                "model": model_name,
                "samples": (
                    low_rul_results[
                        "samples"
                    ]
                ),
                "mae": (
                    low_rul_results[
                        "mae"
                    ]
                ),
                "rmse": (
                    low_rul_results[
                        "rmse"
                    ]
                ),
            }
        )

    low_rul_summary = pd.DataFrame(
        low_rul_summary_rows
    )

    print(
        "\nLow-RUL Model Comparison:"
    )

    print(
        low_rul_summary.to_string(
            index=False
        )
    )

    # =================================================
    # 7. Engine-Level Evaluation
    # =================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ENGINE-LEVEL RUL ERROR ANALYSIS"
    )

    print(
        "=" * 70
    )

    for model_name in [
        "Random Forest",
        "XGBoost",
    ]:

        print(
            f"\n{model_name}"
        )

        engine_metrics, engine_summary = (
            evaluate_engine_level_performance(
                validation_engine_ids,
                y_validation,
                results[
                    model_name
                ]["predictions"],
            )
        )

        print(
            "\nPer-Engine Performance:"
        )

        print(
            engine_metrics.to_string(
                index=False
            )
        )

        print(
            "\nEngine-Level Summary:"
        )

        print(
            f"Mean Engine MAE   : "
            f"{engine_summary['mean_engine_mae']:.4f}"
        )

        print(
            f"Median Engine MAE : "
            f"{engine_summary['median_engine_mae']:.4f}"
        )

        print(
            f"Best Engine MAE   : "
            f"{engine_summary['best_engine_mae']:.4f}"
        )

        print(
            f"Worst Engine MAE  : "
            f"{engine_summary['worst_engine_mae']:.4f}"
        )

    # =================================================
    # 8. Random Forest Feature Importance
    # =================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RANDOM FOREST FEATURE IMPORTANCE"
    )

    print(
        "=" * 70
    )

    feature_importance = analyze_feature_importance(
        random_forest_model,
        X_train.columns,
        top_n=15,
    )

    # =================================================
    # 9. Random Forest Diagnostic Plots
    # =================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RANDOM FOREST PREDICTION ANALYSIS"
    )

    print(
        "=" * 70
    )

    analyze_predictions(
        "Random Forest",
        y_validation,
        results[
            "Random Forest"
        ]["predictions"],
    )


# =====================================================
# Entry Point
# =====================================================

if __name__ == "__main__":
    main()