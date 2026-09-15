import pandas as pd


def calculate_rul(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Remaining Useful Life (RUL) for each engine cycle.

    RUL is calculated independently for each engine unit.
    """

    dataframe = dataframe.copy()

    max_cycles = (
        dataframe
        .groupby("unit_id")["cycle"]
        .max()
        .rename("max_cycle")
    )

    dataframe = dataframe.merge(
        max_cycles,
        on="unit_id",
        how="left",
    )

    dataframe["rul"] = (
        dataframe["max_cycle"] - dataframe["cycle"]
    )

    dataframe.drop(columns=["max_cycle"], inplace=True)

    return dataframe


def create_risk_label(
    dataframe: pd.DataFrame,
    high_risk_threshold: int = 30,
    medium_risk_threshold: int = 75,
) -> pd.DataFrame:
    """
    Create categorical failure-risk labels from RUL.
    """

    dataframe = dataframe.copy()

    dataframe["risk_level"] = pd.cut(
        dataframe["rul"],
        bins=[
            -1,
            high_risk_threshold,
            medium_risk_threshold,
            float("inf"),
        ],
        labels=[
            "high",
            "medium",
            "low",
        ],
    )

    return dataframe


def prepare_training_data(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare C-MAPSS training data with RUL and risk labels.
    """

    dataframe = dataframe.copy()

    dataframe = calculate_rul(dataframe)

    dataframe = create_risk_label(dataframe)

    dataframe = dataframe.sort_values(
        ["unit_id", "cycle"]
    ).reset_index(drop=True)

    return dataframe