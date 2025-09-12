import pandas as pd
from typing import Optional


def reverse_dataframe(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    将一个 Pandas DataFrame 的行进行逆序排列。

    参数:
    df (pd.DataFrame): 原始 DataFrame。

    返回:
    Optional[pd.DataFrame]: 行序相反的新 DataFrame，如果输入不是 DataFrame 则返回 None。
    """
    # 检查输入是否为 DataFrame 类型
    if not isinstance(df, pd.DataFrame):
        print("错误：输入参数必须是一个 pandas.DataFrame 对象。")
        return None

    # 使用 iloc 索引器进行逆序
    reversed_df = df.iloc[::-1]

    return reversed_df


if __name__ == "__main__":
    df = pd.read_csv("./data/demo_600519.csv", index_col=0)
    # remove index
    df.to_csv("./data/demo_600519.csv", index=False)
    df_reverse = reverse_dataframe(df)
    df_reverse.to_csv("./data/demo_600519.csv", index=False)
