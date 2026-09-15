import pandas as pd


def get_sensor_columns(dataframe: pd.DataFrame) -> list[str]:
    """
    Return all sensor columns from the dataframe.
    """

    return [
        column
        for column in dataframe.columns
        if column.startswith("sensor_")
    ]


def find_constant_features(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Find sensor features with no variation.
    """

    sensor_columns = get_sensor_columns(dataframe)

    constant_features = [
        column
        for column in sensor_columns
        if dataframe[column].nunique(dropna=False) <= 1
    ]

    return constant_features


def remove_constant_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove constant sensor features.
    """

    dataframe = dataframe.copy()

    constant_features = find_constant_features(dataframe)

    if constant_features:
        dataframe = dataframe.drop(
            columns=constant_features
        )

    return dataframe