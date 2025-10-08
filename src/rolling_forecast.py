import os
import sys
import torch
import pandas as pd
import argparse


sys.path.insert(0, os.getcwd())
from predictor.stock_predictor import KronosStockPredictor
from typing import Dict, List, Any
from tqdm import tqdm, trange


# CURRENT_DATA_PATH = "./data/stock_data.parquet"
# total_df = pd.read_parquet(CURRENT_DATA_PATH)


def get_stock_data_by_ts_code(df: pd.DataFrame, ts_code: str) -> pd.DataFrame:
    """
    根据 ts_code 筛选股票数据并按日期排序。

    参数:
    df (pd.DataFrame): 包含股票数据的 DataFrame。
    ts_code (str): 想要筛选的股票代码。

    返回:
    pd.DataFrame: 筛选并排序后的股票数据。如果找不到 ts_code 对应的数据，返回一个空的 DataFrame。
    """
    if "code" not in df.columns or "day" not in df.columns:
        print("Error: DataFrame must include 'code' and 'day'.")
        return pd.DataFrame()

    df["day"] = df["day"].astype(str)
    filtered_df = df[df["code"].astype(str) == str(ts_code)].copy()

    if filtered_df.empty:
        print(f"Error: find nothing.")
        return pd.DataFrame()

    # fix index name
    if "vol" in filtered_df.columns:
        filtered_df.rename(columns={"vol": "volume"}, inplace=True)
    elif "volume" not in filtered_df.columns:
        print("Error, no vol or volume found.")

    # filtered_df["timestamps"] = pd.to_datetime(df["day"], format="ISO8601")
    filtered_df["timestamps"] = pd.to_datetime(filtered_df["day"], format="ISO8601")
    return filtered_df.reset_index()


def rolling_forecast(
    ts_code: str, lookback: int, pred_len: int, pred_params=None, **kwargs
) -> pd.DataFrame:
    predictions_list = []
    # 加载股票数据
    date_df = pd.read_csv(f"./data/stock_202510/stock_{ts_code}.csv")
    data_length = len(date_df)

    points_per_day = 48

    if data_length < pred_len + lookback:
        print("Too long prediction windows or too few data")
        return []

    predictor = KronosStockPredictor(data=date_df, **kwargs, stock_code=ts_code)

    # 滑动窗口，以天为单位
    total_windows = (data_length - lookback - pred_len) // points_per_day + 1
    print(f"length of the sliding windows by days: {total_windows}")

    for i in trange(total_windows):
        # i 是按天的起点，换算成 index
        start_pos = i * points_per_day

        pred_df = predictor.predict(
            lookback=lookback,
            pred_len=pred_len,
            pred_params=pred_params,
            start_pos=start_pos,
        )

        if not pred_df.empty:
            predictions_list.append(pred_df)
        else:
            print(f"Error in windows: {i}")

    return predictions_list


def save_rolling_forecasts(
    results: List, base_dir: str = "rolling_forecasts", ts_code=None
):
    """
    将滚动预测结果存储到按股票代码组织的文件夹中，每个预测 DataFrame 保存为一个 csv 文件。
    """
    # print(f"Saving directories into files: {os.path.abspath(base_dir)}")
    os.makedirs(base_dir, exist_ok=True)
    if ts_code is None:
        print("Error! Please fill in the ts_code params")
        exit(0)
    total_length = len(results)
    # for ts_code_index, result_perday in tqdm(enumerate(results), total=total_length):
    # print(f"Saving {ts_code_index} with {len(result_perday)} results")
    save_dir = os.path.join(base_dir, f"202510_{ts_code}")
    os.makedirs(save_dir, exist_ok=True)
    for index, result in enumerate(results):
        file_name = f"index_{index}.csv"
        result = pd.DataFrame(result)
        # print(result.head())
        result.to_csv(os.path.join(save_dir, file_name), index=True)
    # print("All Prediction Done...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", required=True)
    args = parser.parse_args()

    parallel = int(args.parallel)
    # setting GPU environment
    # os.environ["CUDA_VISIBLE_DEVICES"] = f"{parallel-1}"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    print("Current GPU nums:", torch.cuda.device_count())
    # several configs for rolling forecast
    # total stock code count
    TOTAL_STOCK_COUNT = 4963
    MODEL_NAME = "NeoQuasar/Kronos-base"
    TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
    DEVICE = "cuda"
    MAX_CONTENT = 512
    # feat: add parallel
    TARGET_FILE_PATH = "./rolling_forecasts"
    data_processed = [
        int(dir_path.split("_")[-1]) for dir_path in os.listdir(TARGET_FILE_PATH)
    ]
    data_processed = sorted(data_processed)
    all_data = list(range(TOTAL_STOCK_COUNT))
    data_need_to_processed = list(set(all_data) - set(data_processed))

    # chunking for parallel
    length_to_be_processed = len(data_need_to_processed)
    chunk_size = length_to_be_processed // 2
    part1 = data_need_to_processed[0:chunk_size]
    part2 = data_need_to_processed[chunk_size:length_to_be_processed]
    parts = [part1, part2]
    print(f"TSCODE NEED TO BE PROCESSED: {len(parts[parallel-1])}, starting from {parts[parallel-1][0]} to {parts[parallel-1][-1]}")

    for ts_code in parts[parallel - 1]:
        print(f"Starting Forecasting for: {ts_code}")
        result = rolling_forecast(
            ts_code,
            lookback=432,
            pred_len=48,
            model_name=MODEL_NAME,
            max_context=MAX_CONTENT,
            tokenizer_name=TOKENIZER_NAME,
            device=DEVICE,
        )
        save_rolling_forecasts(results=result, ts_code=ts_code)
