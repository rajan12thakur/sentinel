from dataclasses import dataclass

import pandas as pd


@dataclass
class DatasetSplit:
    """
    Container for a machine-learning dataset split.
    """

    X: pd.DataFrame
    y_risk: pd.Series
    y_rul: pd.Series
    engine_ids: pd.Series


def split_features_and_targets(
    dataframe: pd.DataFrame,
) -> DatasetSplit:
    """
    Separate features from prediction targets.

    unit_id is retained separately and is never used
    as a model feature.
    """

    excluded_columns = {
        "unit_id",
        "rul",
        "risk_level",
    }

    feature_columns = [
        column
        for column in dataframe.columns
        if column not in excluded_columns
    ]

    X = dataframe[feature_columns].copy()

    y_risk = dataframe["risk_level"].copy()

    y_rul = dataframe["rul"].copy()

    engine_ids = dataframe["unit_id"].copy()

    return DatasetSplit(
        X=X,
        y_risk=y_risk,
        y_rul=y_rul,
        engine_ids=engine_ids,
    )