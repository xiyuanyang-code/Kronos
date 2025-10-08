# processing for rolling forecast
import pandas as pd
import os
from tqdm import tqdm


def process(ts_code: str):
    loader_path = os.path.join(LOADER_FILE_PATH, f"stock_{ts_code}.csv")
    try:
        # Load original (true) data
        original_data = pd.read_csv(loader_path)
        original_data["timestamps"] = pd.to_datetime(original_data["timestamps"])
        original_data = original_data.set_index("timestamps")

        # Select and rename the required true columns
        true_cols = ["open", "high", "low", "close"]
        original_data = original_data[true_cols].rename(
            columns={col: f"true_{col}" for col in true_cols}
        )
        # print(f"Loaded original data head for {ts_code}:\n{original_data.head()}")

    except FileNotFoundError:
        print(f"Error: Original data file not found at {loader_path}")
        return

    target_dir_path = os.path.join(TARGET_FILE_PATH, f"202510_{ts_code}")
    output_dir_path = os.path.join(OUTPUT_FILE_PATH, f"202510_{ts_code}")
    if os.path.exists(output_dir_path):
        print(f"Output file path: {output_dir_path} exists, skipping")
        return
    os.makedirs(output_dir_path, exist_ok=True)

    try:
        for file_name in tqdm(os.listdir(target_dir_path)):
            if file_name.endswith(".csv"):
                target_file_path = os.path.join(target_dir_path, file_name)

                # Load forecast data
                target_df = pd.read_csv(target_file_path)
                target_df["timestamps"] = pd.to_datetime(target_df["timestamps"])
                target_df = target_df.set_index("timestamps")

                # Rename predicted OHLC columns for clarity
                predicted_cols = ["open", "high", "low", "close"]
                target_df = target_df.rename(
                    columns={
                        col: f"predicted_{col}"
                        for col in predicted_cols
                        if col in target_df.columns
                    }
                )

                # print(f"Loaded target data head for {file_name}:\n{target_df.head()}")

                # Merge the true values into the forecast DataFrame by timestamp index
                merged_df = pd.merge(
                    target_df,
                    original_data,
                    left_index=True,
                    right_index=True,
                    how="inner",
                )

                # Save Merged Result
                merged_df = merged_df.reset_index()
                output_file_path = os.path.join(output_dir_path, file_name)

                merged_df.to_csv(output_file_path, index=False)
                # print(f"Merged Data head:\n{merged_df.head()}")
                # print("-" * 50)

    except FileNotFoundError:
        print(f"Error: Target directory not found at {target_dir_path}")
    except Exception as e:
        print(f"An error occurred while processing {ts_code}: {e}")


if __name__ == "__main__":
    TARGET_FILE_PATH = "./rolling_forecasts"
    OUTPUT_FILE_PATH = "./rolling_forecasts_output"
    LOADER_FILE_PATH = "./data/stock_202510"
    os.makedirs(OUTPUT_FILE_PATH, exist_ok=True)
    data_processed = [
        int(dir_path.split("_")[-1]) for dir_path in os.listdir(TARGET_FILE_PATH)
    ]
    data_processed = sorted(data_processed)
    for ts_code in data_processed:
        process(ts_code=ts_code)
