import pandas as pd

from ml.utils.constants import CMAPSS_COLUMNS


def validate_cmapss_dataframe(dataframe: pd.DataFrame) -> None:
    """
    Validate the basic structural integrity of a C-MAPSS dataframe.
    """

    if dataframe.empty:
        raise ValueError("C-MAPSS dataframe is empty.")

    missing_columns = set(CMAPSS_COLUMNS) - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {sorted(missing_columns)}"
        )

    if dataframe["unit_id"].isna().any():
        raise ValueError("unit_id contains missing values.")

    if dataframe["cycle"].isna().any():
        raise ValueError("cycle contains missing values.")

    if (dataframe["cycle"] <= 0).any():
        raise ValueError("cycle must contain positive values.")

    if dataframe["unit_id"].nunique() == 0:
        raise ValueError("No engine units found.")

    sensor_columns = [
        column
        for column in dataframe.columns
        if column.startswith("sensor_")
    ]

    if dataframe[sensor_columns].isna().any().any():
        raise ValueError("Sensor data contains missing values.")