from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_classifier(
    model: Any,
    X,
    y,
    class_order: list[str],
) -> dict:
    """
    Evaluate a multiclass classification model.

    Metrics include:
    - Accuracy
    - Macro precision
    - Macro recall
    - Macro F1
    - Weighted F1
    - Multiclass PR-AUC
    - Confusion matrix
    - Classification report
    """

    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions,
    )

    precision = precision_score(
        y,
        predictions,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y,
        predictions,
        average="weighted",
        zero_division=0,
    )

    probabilities = model.predict_proba(X)

    y_encoded = np.array(
        [
            class_order.index(label)
            for label in y
        ]
    )

    pr_auc = average_precision_score(
        y_encoded,
        probabilities,
        average="macro",
    )

    matrix = confusion_matrix(
        y,
        predictions,
        labels=class_order,
    )

    report = classification_report(
        y,
        predictions,
        labels=class_order,
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": macro_f1,
        "f1_weighted": weighted_f1,
        "pr_auc_macro": pr_auc,
        "confusion_matrix": matrix,
        "classification_report": report,
    }


def print_classification_results(
    model_name: str,
    results: dict,
    class_order: list[str],
) -> None:
    """
    Print classification evaluation results.
    """

    print(
        f"\n{'=' * 60}"
    )

    print(
        f"{model_name}"
    )

    print(
        f"{'=' * 60}"
    )

    print(
        f"Accuracy       : "
        f"{results['accuracy']:.4f}"
    )

    print(
        f"Macro Precision: "
        f"{results['precision_macro']:.4f}"
    )

    print(
        f"Macro Recall   : "
        f"{results['recall_macro']:.4f}"
    )

    print(
        f"Macro F1       : "
        f"{results['f1_macro']:.4f}"
    )

    print(
        f"Weighted F1    : "
        f"{results['f1_weighted']:.4f}"
    )

    print(
        f"Macro PR-AUC   : "
        f"{results['pr_auc_macro']:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        f"Classes: {class_order}"
    )

    print(
        results["confusion_matrix"]
    )

    print("\nClassification Report:")

    print(
        results["classification_report"]
    )