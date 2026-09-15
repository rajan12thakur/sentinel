import pandas as pd

from sklearn.model_selection import GroupShuffleSplit


def split_by_engine(
    dataframe: pd.DataFrame,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split C-MAPSS data by engine unit.

    All observations belonging to the same engine remain
    in the same split.
    """

    if "unit_id" not in dataframe.columns:
        raise ValueError(
            "Dataframe must contain 'unit_id'."
        )

    if not 0 < validation_size < 1:
        raise ValueError(
            "validation_size must be between 0 and 1."
        )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_size,
        random_state=random_state,
    )

    train_indices, validation_indices = next(
        splitter.split(
            dataframe,
            groups=dataframe["unit_id"],
        )
    )

    train_df = (
        dataframe
        .iloc[train_indices]
        .copy()
        .reset_index(drop=True)
    )

    validation_df = (
        dataframe
        .iloc[validation_indices]
        .copy()
        .reset_index(drop=True)
    )

    return train_df, validation_df

def validate_engine_split(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> None:
    """
    Verify that no engine appears in both splits.
    """

    train_engines = set(train_df["unit_id"].unique())
    validation_engines = set(
        validation_df["unit_id"].unique()
    )

    overlap = train_engines.intersection(
        validation_engines
    )

    if overlap:
        raise ValueError(
            f"Engine leakage detected: {sorted(overlap)}"
        )