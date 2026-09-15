from typing import Any

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_regressor(
    model: Any,
    X,
    y,
) -> dict:
    """
    Evaluate a regression model.

    Metrics include:
    - MAE
    - RMSE
    - R²
    - Median Absolute Error
    - Maximum Absolute Error
    """

    predictions = model.predict(X)

    absolute_errors = abs(y - predictions)

    mae = mean_absolute_error(
        y,
        predictions,
    )

    rmse = mean_squared_error(
    y,
    predictions,
    ) ** 0.5

    r2 = r2_score(
        y,
        predictions,
    )

    median_absolute_error = absolute_errors.median()

    maximum_absolute_error = absolute_errors.max()

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "median_absolute_error": median_absolute_error,
        "maximum_absolute_error": maximum_absolute_error,
        "predictions": predictions,
    }


def print_regression_results(
    model_name: str,
    results: dict,
) -> None:
    """
    Print regression evaluation results.
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
        f"MAE                    : "
        f"{results['mae']:.4f}"
    )

    print(
        f"RMSE                   : "
        f"{results['rmse']:.4f}"
    )

    print(
        f"R²                     : "
        f"{results['r2']:.4f}"
    )

    print(
        f"Median Absolute Error  : "
        f"{results['median_absolute_error']:.4f}"
    )

    print(
        f"Maximum Absolute Error : "
        f"{results['maximum_absolute_error']:.4f}"
    )