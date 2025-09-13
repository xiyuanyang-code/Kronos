import os
import sys
import torch
import pandas as pd

sys.path.insert(0, os.getcwd())
from predictor.stock_predictor import KronosStockPredictor
from typing import Dict, List

CURRENT_DATA_PATH = "./data/stock_data.parquet"
total_df = pd.read_parquet(CURRENT_DATA_PATH)
print(total_df.head(10))


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

    filtered_df = df[df["code"].astype(str) == str(ts_code)].copy()

    if filtered_df.empty:
        print(f"Error: find nothing.")
        return pd.DataFrame()

    filtered_df["day"] = pd.to_datetime(filtered_df["day"], format="%Y%m%d")
    sorted_df = filtered_df.sort_values(by="day", ascending=True)
    return sorted_df


def rolling_forecast(ts_code: str) -> pd.DataFrame:
    if "." in ts_code:
        ts_code = ts_code.split(".")[0]
    date_df = get_stock_data_by_ts_code(df=total_df, ts_code=ts_code)
    print(f"Length of daily data for tscode `{ts_code}`: {len(date_df)}")
    pass


if __name__ == "__main__":
    print("Start rolling forecast.")
    # setting GPU environment
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
    print(torch.cuda.device_count())

    # several configs for rolling forecast

    # trade codes
    # todo add more ts_codes
    TRADE_CODES = ["600372", "600197", "600867"]
    # * **600372**: **中航机载** (全称：中航航空电子系统股份有限公司)
    # * **600197**: **伊力特** (全称：新疆伊力特实业股份有限公司)
    # * **600867**: **通化东宝** (全称：通化东宝药业股份有限公司)

    results = [rolling_forecast(ts_code) for ts_code in TRADE_CODES]
