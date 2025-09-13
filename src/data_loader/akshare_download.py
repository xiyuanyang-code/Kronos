import akshare as ak
import pandas as pd


def get_5min_data_akshare(
    symbol="600519", start_date="2025-01-01", end_date="2025-07-31"
):
    """
    使用 Akshare 获取A股5分钟K线数据。
    Args:
        symbol (str): 股票代码，例如 "600519"。
        start_date (str): 开始日期，格式为 "YYYYMMDD"。
        end_date (str): 结束日期，格式为 "YYYYMMDD"。
    Returns:
        pd.DataFrame: 包含5分钟K线数据的DataFrame。
    """
    df = ak.stock_zh_a_hist_min_em(
        symbol=symbol,
        period="5",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq",
    )

    df.to_csv(f"./data/min_{symbol}_akshare_demo.csv", index=False)
    return df


if __name__ == "__main__":
    # 获取贵州茅台（600519）2024年上半年的5分钟数据
    stock_data = get_5min_data_akshare()

    # 打印数据的前5行
    print(stock_data.head())

    # 打印获取到的数据量
    print(f"获取到的数据量：{len(stock_data)}条")

    test_df = ak.stock_zh_a_hist_min_em()
    # print(test_df)
