import os
import pandas as pd


# if __name__ == "__main__":
#     data_path = "./data/stock_202510/stock_1.csv"
#     df = pd.read_csv(data_path)

#     # 把 timestamps 转换成 datetime
#     df["timestamps"] = pd.to_datetime(df["timestamps"])

#     # 提取日期部分
#     df["date"] = df["timestamps"].dt.date

#     # 统计每天的行数
#     daily_counts = df.groupby("date").size().reset_index(name="row_count")

#     print(daily_counts)

#     # 如果你想保存结果到 csv
#     daily_counts.to_csv("./data/daily_counts.csv", index=False)


import pandas as pd

# 读取数据
df = pd.read_csv("./data/stock_202510/stock_0.csv")

# 转换时间
df["timestamps"] = pd.to_datetime(df["timestamps"])

# 找到 2021-01-15 的数据并替换
mask = df["timestamps"].dt.date == pd.to_datetime("2021-01-15").date()
df.loc[mask, df.columns != "timestamps"] = 0

# 保存回文件
df.to_csv("./data/stock_202510/stock_0.csv", index=False)
