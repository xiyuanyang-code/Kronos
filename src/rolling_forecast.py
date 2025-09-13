import os
import sys
import torch
import pandas as pd


sys.path.insert(0, os.getcwd())
from predictor.stock_predictor import KronosStockPredictor
from typing import Dict, List, Any
from tqdm import tqdm, trange


CURRENT_DATA_PATH = "./data/stock_data.parquet"
total_df = pd.read_parquet(CURRENT_DATA_PATH)


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

    filtered_df["timestamps"] = pd.to_datetime(df["day"], format="ISO8601")
    return filtered_df.reset_index()


def rolling_forecast(
    ts_code: str, lookback, pred_len, pred_params=None, **kwargs
) -> pd.DataFrame:
    predictions_list = []
    if "." in ts_code:
        ts_code = ts_code.split(".")[0]
    date_df = get_stock_data_by_ts_code(df=total_df, ts_code=ts_code)
    data_length = len(date_df)

    if data_length < pred_len + lookback:
        print("Too long prediction windows or too few data")
        return []

    predictor = KronosStockPredictor(data=date_df, **kwargs, stock_code=ts_code)

    # start sliding window prediction
    print(f"length of the sliding windows: {data_length - lookback - pred_len + 1}")
    for i in trange(data_length - lookback - pred_len + 1):
        # i is starting index
        pred_df = predictor.predict(
            lookback=lookback, pred_len=pred_len, pred_params=pred_params, start_pos=i
        )

        if not pred_df.empty:
            predictions_list.append(pred_df)
        else:
            print(f"Error in windows: {i}")

    return predictions_list


def save_rolling_forecasts(
    results: List[Dict[str, List[pd.DataFrame]]], base_dir: str = "rolling_forecasts"
):
    """
    将滚动预测结果存储到按股票代码组织的文件夹中，每个预测 DataFrame 保存为一个 Parquet 文件。

    Args:
        results (List[Dict[str, List[pd.DataFrame]]]):
            包含滚动预测结果的列表，每个元素是一个字典，键为股票代码，值为预测结果的列表。
        base_dir (str):
            用于存储所有预测结果的根目录名称。
    """
    print(f"正在将预测结果存储到目录: {os.path.abspath(base_dir)}")

    os.makedirs(base_dir, exist_ok=True)
    total_length = len(results)
    for result_dict in tqdm(results, total=total_length):
        ts_code = list(result_dict.keys())[0]
        pred_dfs = result_dict[ts_code]

        stock_dir = os.path.join(base_dir, ts_code)
        os.makedirs(stock_dir, exist_ok=True)

        print(f"\n正在存储股票 {ts_code} 的 {len(pred_dfs)} 个预测结果...")

        # 遍历每个预测 DataFrame 并保存为 Parquet 文件
        for i, df in enumerate(pred_dfs):
            # 为了方便命名，命名采取预测窗口的第一天的日期
            time_pred_ini = str(df["timestamps"].iloc[0]).strip()
            file_path = os.path.join(
                stock_dir, f"prediction_{time_pred_ini}_{i}.parquet"
            )

            try:
                df.to_parquet(file_path, index=False)
                # print(f"  - 已保存: {file_path}")
            except Exception as e:
                print(f"  - 保存文件 {file_path} 失败: {e}")
            exit(0)

    print("\n所有预测结果存储完成。")


if __name__ == "__main__":
    print("Start rolling forecast.")
    # setting GPU environment
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
    print("Current GPU nums:", torch.cuda.device_count())

    # several configs for rolling forecast

    # trade codes
    # todo add more ts_codes
    TRADE_CODES = ["600372", "600197", "600867"]
    # * **600372**: **中航机载** (全称：中航航空电子系统股份有限公司)
    # * **600197**: **伊力特** (全称：新疆伊力特实业股份有限公司)
    # * **600867**: **通化东宝** (全称：通化东宝药业股份有限公司)

    MODEL_NAME = "NeoQuasar/Kronos-base"
    TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
    DEVICE = "cuda"
    MAX_CONTENT = 512

    results = [
        {
            ts_code: rolling_forecast(
                ts_code,
                lookback=400,
                pred_len=11,
                model_name=MODEL_NAME,
                max_context=MAX_CONTENT,
                tokenizer_name=TOKENIZER_NAME,
                device=DEVICE,
            )
        }
        for ts_code in TRADE_CODES
    ]
