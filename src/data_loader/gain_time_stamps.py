import numpy as np
import pandas as pd
import os

from tqdm import tqdm


def gain_time_stamps():
    # 读取parquet文件中的day列
    df_time_day = pd.read_parquet("./data/stock_data.parquet", columns=["day"])
    # 对day 列进行去重
    df_time_day = df_time_day.drop_duplicates(subset=["day"])
    # 提取2021年以后2024年以前的数据
    df_time_day = df_time_day[
        (df_time_day["day"] >= 20210101) & (df_time_day["day"] < 20240101)
    ]
    # 转为时间戳格式
    df_time_day["day"] = pd.to_datetime(df_time_day["day"], format="%Y%m%d")
    # print(df_time_day.head())
    # 在时间戳中增加小时和分钟 五分钟间隔 从9:30到11:30
    time_stamps = []
    for day in tqdm(df_time_day["day"], total=len(df_time_day["day"])):
        for hour in range(9, 12):
            for minute in range(0, 60, 5):
                if hour == 9 and minute <= 30:
                    continue
                if hour == 11 and minute > 30:
                    continue
                time_stamps.append(day.replace(hour=hour, minute=minute))
        for hour in range(13, 15):
            for minute in range(0, 60, 5):
                if hour == 13 and minute == 0:
                    continue
                time_stamps.append(day.replace(hour=hour, minute=minute))
        time_stamps.append(day.replace(hour=15, minute=0))
    return time_stamps


def load_data(file_path):
    data = np.load(file_path)
    return data


def process_stock_data(index):
    data_s = data[:, index, :]
    # 把data_s转为一维数组
    data_s = data_s.flatten()
    # 取和time_stamps长度相同的数据
    data_s = data_s[: len(time_stamps)]
    out_data = {
        "timestamps": time_stamps,
        "open": data_s,
        "high": data_s,
        "low": data_s,
        "close": data_s,
    }
    return out_data


def process_centain_stock_data():
    n = data.shape[1]
    for index in tqdm(range(n), total=n):
        out_data = process_stock_data(index)
        df = pd.DataFrame(out_data)
        df.to_csv(
            f"./data/stock_202510/stock_{index}.csv", index=False, encoding="utf-8"
        )


if __name__ == "__main__":
    os.makedirs("./data/stock_202510", exist_ok=True)
    time_stamps = gain_time_stamps()
    print("The length of the stock data:", len(time_stamps))

    file_path = "./data/train_X.npy"
    print("Loading data...")
    data = load_data(file_path)
    print("Process stock data...")
    process_centain_stock_data()
