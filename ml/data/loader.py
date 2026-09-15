from pathlib import Path

import pandas as pd

from ml.utils.constants import CMAPSS_COLUMNS


def load_cmapss_train(file_path: str | Path) -> pd.DataFrame:
    """
    Load a C-MAPSS training dataset.

    Parameters
    ----------
    file_path:
        Path to train_FD001.txt.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing engine cycles, operating settings,
        and sensor measurements.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"C-MAPSS training file not found: {file_path}"
        )

    dataframe = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=CMAPSS_COLUMNS,
    )

    return dataframe