from pathlib import Path

import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.evaluation.classification import (
    evaluate_classifier,
    print_classification_results,
)


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


TARGET_COLUMN = "risk_level"

ID_COLUMN = "unit_id"

CLASS_ORDER = [
    "high",
    "medium",
    "low",
]


def load_datasets():
    """
    Load prepared training and validation datasets.
    """

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    validation_df = pd.read_csv(
        VALIDATION_FILE
    )

    feature_columns = [
        column
        for column in train_df.columns
        if column not in {
            ID_COLUMN,
            "rul",
            TARGET_COLUMN,
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

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
    )


def main():
    print(
        "Loading prepared datasets..."
    )

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
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
        "\nTraining class distribution:"
    )

    print(
        y_train.value_counts()
    )

    print(
        "\nValidation class distribution:"
    )

    print(
        y_validation.value_counts()
    )

    results = {}

    # -------------------------------------------------
    # 1. Dummy baseline
    # -------------------------------------------------

    print(
        "\nTraining DummyClassifier..."
    )

    dummy_model = DummyClassifier(
        strategy="most_frequent"
    )

    dummy_model.fit(
        X_train,
        y_train,
    )

    dummy_results = evaluate_classifier(
        dummy_model,
        X_validation,
        y_validation,
        CLASS_ORDER,
    )

    results["Dummy"] = dummy_results

    print_classification_results(
        "Dummy Baseline",
        dummy_results,
        CLASS_ORDER,
    )

    # -------------------------------------------------
    # 2. Logistic Regression
    # -------------------------------------------------

    print(
        "\nTraining Logistic Regression..."
    )

    logistic_model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    logistic_model.fit(
        X_train,
        y_train,
    )

    logistic_results = evaluate_classifier(
        logistic_model,
        X_validation,
        y_validation,
        CLASS_ORDER,
    )

    results["Logistic Regression"] = (
        logistic_results
    )

    print_classification_results(
        "Logistic Regression",
        logistic_results,
        CLASS_ORDER,
    )

    # -------------------------------------------------
    # 3. Random Forest
    # -------------------------------------------------

    print(
        "\nTraining Random Forest..."
    )

    random_forest_model = (
        RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    )

    random_forest_model.fit(
        X_train,
        y_train,
    )

    random_forest_results = (
        evaluate_classifier(
            random_forest_model,
            X_validation,
            y_validation,
            CLASS_ORDER,
        )
    )

    results["Random Forest"] = (
        random_forest_results
    )

    print_classification_results(
        "Random Forest",
        random_forest_results,
        CLASS_ORDER,
    )

    # -------------------------------------------------
    # 4. Summary
    # -------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "MODEL COMPARISON"
    )

    print(
        "=" * 60
    )

    summary_rows = []

    for model_name, result in results.items():

        summary_rows.append(
            {
                "model": model_name,
                "accuracy": result[
                    "accuracy"
                ],
                "precision_macro": result[
                    "precision_macro"
                ],
                "recall_macro": result[
                    "recall_macro"
                ],
                "f1_macro": result[
                    "f1_macro"
                ],
                "pr_auc_macro": result[
                    "pr_auc_macro"
                ],
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


if __name__ == "__main__":
    main()