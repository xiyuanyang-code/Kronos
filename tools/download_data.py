import os
import sys
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime
from utils.initializer import ComponentInitializer

def main():
    # 初始化组件
    init = ComponentInitializer()
    config = init.load_config()
    downloader = init.init_downloader()
    preprocessor = init.init_preprocessor()
    data_path = config.get('data_path', './data')  # 从配置获取data_path

    # 设置命令行参数
    parser = argparse.ArgumentParser(description='股票数据下载工具')
    parser.add_argument('--no_update', action='store_true',
                       help='不更新股票日线行情')
    parser.add_argument('--start_date', type=str, default='20140101',
                       help='起始日期')
    parser.add_argument('--end_date', type=str, default=datetime.now().strftime('%Y%m%d'),
                       help='结束日期')
    parser.add_argument('--output', type=str, default=os.path.join(data_path, 'stock_data.parquet'),
                       help='导出文件路径')
    args = parser.parse_args()

    if not args.no_update:
        print(f"开始更新股票日线行情，日期范围：{args.start_date} 到 {args.end_date}...")
        downloader.update_by_trade_dates(start_date=args.start_date, end_date=args.end_date)

    print(f"开始导出股票数据到{args.output}...")
    df = downloader.export_daily_to_parquet(
        code_prefixes=config['data_config']['code_prefixes'],
        start_date=args.start_date,
        end_date=args.end_date
    )
    print('正在保存数据...', end='', flush=True)
    df.to_parquet(args.output, index=False)
    print("\r股票数据导出完成！", flush=True)

if __name__ == '__main__':
    main()