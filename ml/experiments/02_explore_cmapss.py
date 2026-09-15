from ml.data.loader import load_cmapss_train
from ml.data.validator import validate_cmapss_dataframe
from ml.data.preprocessing import prepare_training_data
from ml.features.engineering import build_features
from ml.utils.constants import CMAPSS_DIR


def main():
    train_file = CMAPSS_DIR / "train_FD001.txt"

    print("Loading dataset...")

    df = load_cmapss_train(train_file)

    print("\n========== DATASET OVERVIEW ==========")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Engine units: {df['unit_id'].nunique()}")
    print(
        f"Cycle range: "
        f"{df['cycle'].min()} - {df['cycle'].max()}"
    )

    print("\n========== COLUMN TYPES ==========")
    print(df.dtypes)

    print("\n========== MISSING VALUES ==========")
    missing = df.isna().sum()

    print(
        missing[missing > 0]
        if missing.any()
        else "No missing values found."
    )

    print("\n========== ENGINE LENGTH ==========")

    engine_lengths = (
        df.groupby("unit_id")["cycle"]
        .max()
    )

    print(engine_lengths.describe())

    print("\n========== SENSOR VARIANCE ==========")

    sensor_columns = [
        column
        for column in df.columns
        if column.startswith("sensor_")
    ]

    sensor_variance = df[sensor_columns].var()

    print(
        sensor_variance
        .sort_values()
    )

    print("\n========== CONSTANT SENSORS ==========")

    constant_sensors = [
        sensor
        for sensor in sensor_columns
        if df[sensor].nunique() <= 1
    ]

    if constant_sensors:
        print(constant_sensors)
    else:
        print("No constant sensors found.")

    print("\n========== PREPARE TARGET ==========")

    validate_cmapss_dataframe(df)

    df = prepare_training_data(df)

    print(
        "\nRUL statistics:"
    )

    print(df["rul"].describe())

    print("\nRisk distribution:")

    print(
        df["risk_level"]
        .value_counts()
        .sort_index()
    )

    print("\n========== CORRELATION WITH RUL ==========")

    correlations = (
        df[sensor_columns + ["rul"]]
        .corr()["rul"]
        .drop("rul")
        .sort_values()
    )

    print(correlations)

    print("\nEDA completed successfully.")


if __name__ == "__main__":
    main()