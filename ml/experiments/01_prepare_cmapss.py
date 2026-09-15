from ml.data.loader import load_cmapss_train
from ml.data.validator import validate_cmapss_dataframe
from ml.data.preprocessing import prepare_training_data
from ml.features.engineering import build_features
from ml.utils.constants import CMAPSS_DIR


def main():
    train_file = CMAPSS_DIR / "train_FD001.txt"

    print("Loading C-MAPSS training data...")

    dataframe = load_cmapss_train(train_file)

    print(f"Raw shape: {dataframe.shape}")

    print("Validating dataset...")

    validate_cmapss_dataframe(dataframe)

    print("Preparing training targets...")

    dataframe = prepare_training_data(dataframe)

    print("Building features...")

    dataframe = build_features(dataframe)

    print(f"Processed shape: {dataframe.shape}")

    print("\nRUL statistics:")
    print(dataframe["rul"].describe())

    print("\nRisk distribution:")
    print(dataframe["risk_level"].value_counts())

    print("\nSample:")
    print(dataframe.head())


if __name__ == "__main__":
    main()