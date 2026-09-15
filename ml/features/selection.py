import pandas as pd


def find_constant_features(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Find features with no variation.
    """

    constant_features = [
        column
        for column in dataframe.columns
        if dataframe[column].nunique(dropna=False) <= 1
    ]

    return constant_features


def remove_constant_features(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove features with no variation.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        The cleaned dataframe and the list of removed features.
    """

    dataframe = dataframe.copy()

    constant_features = find_constant_features(
        dataframe
    )

    if constant_features:
        dataframe = dataframe.drop(
            columns=constant_features
        )

    return dataframe, constant_features