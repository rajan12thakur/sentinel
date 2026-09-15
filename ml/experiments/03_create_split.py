from ml.data.loader import load_cmapss_train
from ml.data.preprocessing import prepare_training_data
from ml.data.splitting import (
    split_by_engine,
    validate_engine_split,
)
from ml.utils.constants import CMAPSS_DIR


def main():
    train_file = CMAPSS_DIR / "train_FD001.txt"

    print("Loading dataset...")

    df = load_cmapss_train(train_file)

    print("Preparing targets...")

    df = prepare_training_data(df)

    print("Creating engine-aware split...")

    train_df, validation_df = split_by_engine(
        df,
        validation_size=0.2,
        random_state=42,
    )

    print("\n========== SPLIT SUMMARY ==========")

    print(
        f"Training rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_df):,}"
    )

    print(
        f"Training engines: "
        f"{train_df['unit_id'].nunique()}"
    )

    print(
        f"Validation engines: "
        f"{validation_df['unit_id'].nunique()}"
    )

    validate_engine_split(
        train_df,
        validation_df,
    )

    print("\nEngine leakage check: PASSED")

    print("\nTraining engines:")
    print(
        sorted(train_df["unit_id"].unique())
    )

    print("\nValidation engines:")
    print(
        sorted(validation_df["unit_id"].unique())
    )


if __name__ == "__main__":
    main()