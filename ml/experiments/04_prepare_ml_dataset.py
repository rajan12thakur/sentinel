from ml.data.dataset import split_features_and_targets
from ml.data.loader import load_cmapss_train
from ml.data.preprocessing import prepare_training_data
from ml.data.splitting import (
    split_by_engine,
    validate_engine_split,
)
from ml.features.engineering import build_features
from ml.features.selection import remove_constant_features
from ml.utils.constants import (
    CMAPSS_DIR,
    PROCESSED_DATA_DIR,
)


def main():
    train_file = CMAPSS_DIR / "train_FD001.txt"

    print("Loading C-MAPSS dataset...")

    df = load_cmapss_train(train_file)

    print("Preparing targets...")

    df = prepare_training_data(df)

    print("Creating engine-aware split...")

    train_df, validation_df = split_by_engine(
        df,
        validation_size=0.2,
        random_state=42,
    )

    validate_engine_split(
        train_df,
        validation_df,
    )

    print("Building training features...")

    train_df = build_features(train_df)

    print("Building validation features...")

    validation_df = build_features(validation_df)

    print("Separating features and targets...")

    train_data = split_features_and_targets(
        train_df
    )

    validation_data = split_features_and_targets(
        validation_df
    )

    print("Removing constant features...")

    train_X, removed_features = remove_constant_features(
        train_data.X
    )

    validation_X = validation_data.X.drop(
        columns=removed_features,
        errors="ignore",
    )

    # Reconstruct processed datasets with targets.
    train_processed = train_X.copy()

    train_processed["rul"] = train_data.y_rul.values
    train_processed["risk_level"] = train_data.y_risk.values
    train_processed["unit_id"] = train_data.engine_ids.values

    validation_processed = validation_X.copy()

    validation_processed["rul"] = (
        validation_data.y_rul.values
    )

    validation_processed["risk_level"] = (
        validation_data.y_risk.values
    )

    validation_processed["unit_id"] = (
        validation_data.engine_ids.values
    )

    # Keep identifier and targets at the beginning/end
    # in a predictable order.
    train_processed = train_processed[
        ["unit_id"]
        + list(train_X.columns)
        + ["rul", "risk_level"]
    ]

    validation_processed = validation_processed[
        ["unit_id"]
        + list(validation_X.columns)
        + ["rul", "risk_level"]
    ]

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_output = PROCESSED_DATA_DIR / "train.csv"
    validation_output = (
        PROCESSED_DATA_DIR / "validation.csv"
    )

    train_processed.to_csv(
        train_output,
        index=False,
    )

    validation_processed.to_csv(
        validation_output,
        index=False,
    )

    print("\n========== ML DATASET ==========")

    print(
        f"Training rows: {len(train_X):,}"
    )

    print(
        f"Validation rows: {len(validation_X):,}"
    )

    print(
        f"Training features: {train_X.shape[1]}"
    )

    print(
        f"Validation features: {validation_X.shape[1]}"
    )

    print(
        f"Removed constant features: "
        f"{len(removed_features)}"
    )

    print("\nRemoved features:")

    for feature in removed_features:
        print(f"  - {feature}")

    print("\nRisk distribution — training:")

    print(
        train_data.y_risk.value_counts()
        .sort_index()
    )

    print("\nRisk distribution — validation:")

    print(
        validation_data.y_risk.value_counts()
        .sort_index()
    )

    print("\nRUL distribution — training:")

    print(
        train_data.y_rul.describe()
    )

    print("\nRUL distribution — validation:")

    print(
        validation_data.y_rul.describe()
    )

    print("\nEngine leakage check: PASSED")

    print("\nSaved datasets:")

    print(f"  - {train_output}")
    print(f"  - {validation_output}")

    print(
        "\nML dataset preparation completed."
    )


if __name__ == "__main__":
    main()