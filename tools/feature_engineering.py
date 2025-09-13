import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
from utils.initializer import ComponentInitializer
from utils.data_preprocessor import DataPreprocessor


def main():
    # 初始化组件
    init = ComponentInitializer()
    config = init.load_config()  # 通过initializer获取配置
    data_path = config.get("data_path", "./data")  # 从配置获取data_path

    preprocessor = init.init_preprocessor()

    # 设置命令行参数
    parser = argparse.ArgumentParser(description="股票特征工程工具")
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(data_path, "stock_data.parquet"),
        help=f"输入文件路径，默认{data_path}/stock_data.parquet",
    )
    parser.add_argument("--output", type=str, help="输出文件路径，默认根据方法自动生成")
    parser.add_argument("--factor", type=str, default="adj", help="因子名称，默认为adj")
    parser.add_argument(
        "--norm", type=str, default="robust_norm", help="标准化方法，默认为robust_norm"
    )
    parser.add_argument(
        "--filt",
        type=int,
        default=1,
        help="是否过滤异常数据，0不过滤，1过滤nan，2过滤inf，3过滤nan和inf",
    )
    parser.add_argument(
        "--fit_start",
        type=str,
        default=20160101,
        help="拟合开始日期，格式YYYYMMDD，默认为20160101",
    )
    parser.add_argument(
        "--fit_end",
        type=str,
        default=20201231,
        help="拟合结束日期，格式YYYYMMDD，默认为20201231",
    )
    args = parser.parse_args()

    # 设置默认输出路径
    if not args.output:
        args.output = os.path.join(data_path, f"stock_data_{args.factor}.parquet")

    # 读取数据
    print(f"正在从 {args.input} 加载数据...")
    df = pd.read_parquet(args.input)

    # 应用特征工程
    print(f"开始应用 {args.factor} 因子...")
    df = preprocessor.apply_factor(df, args.factor)
    df["SecurityID"] = df["code"]
    df["time"] = df["day"]

    # 标准化
    if args.norm == "robust_norm":
        print("正在进行Robust标准化...")
        df = preprocessor.robust_norm(
            df, fit_start=args.fit_start, fit_end=args.fit_end, filt=args.filt
        )

    # 保存结果
    print(f"正在保存结果到 {args.output}...")
    df.to_parquet(args.output, index=False)

    print(f"特征工程完成，结果已保存到 {args.output}")


if __name__ == "__main__":
    main()
