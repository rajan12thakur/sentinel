import pandas as pd


def get_raw_sensor_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Return only the original C-MAPSS sensor columns.
    """

    return [
        column
        for column in dataframe.columns
        if column.startswith("sensor_")
        and "_rolling_" not in column
        and "_delta" not in column
    ]


def add_rolling_features(
    dataframe: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """
    Add rolling mean and rolling standard deviation
    for the original sensor measurements.
    """

    dataframe = dataframe.copy()

    sensor_columns = get_raw_sensor_columns(dataframe)

    dataframe = dataframe.sort_values(
        ["unit_id", "cycle"]
    )

    grouped = dataframe.groupby(
        "unit_id",
        group_keys=False,
    )

    for sensor in sensor_columns:

        dataframe[f"{sensor}_rolling_mean"] = (
            grouped[sensor]
            .rolling(
                window=window,
                min_periods=1,
            )
            .mean()
            .reset_index(
                level=0,
                drop=True,
            )
        )

        dataframe[f"{sensor}_rolling_std"] = (
            grouped[sensor]
            .rolling(
                window=window,
                min_periods=2,
            )
            .std()
            .reset_index(
                level=0,
                drop=True,
            )
            .fillna(0)
        )

    return dataframe.reset_index(drop=True)


def add_delta_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add cycle-to-cycle changes for the original sensors.
    """

    dataframe = dataframe.copy()

    sensor_columns = get_raw_sensor_columns(dataframe)

    dataframe = dataframe.sort_values(
        ["unit_id", "cycle"]
    )

    for sensor in sensor_columns:
        dataframe[f"{sensor}_delta"] = (
            dataframe
            .groupby("unit_id")[sensor]
            .diff()
            .fillna(0)
        )

    return dataframe.reset_index(drop=True)


def build_features(
    dataframe: pd.DataFrame,
    rolling_window: int = 5,
) -> pd.DataFrame:
    """
    Build the complete feature set.
    """

    dataframe = dataframe.copy()

    dataframe = add_rolling_features(
        dataframe,
        window=rolling_window,
    )

    dataframe = add_delta_features(
        dataframe
    )

    return dataframe