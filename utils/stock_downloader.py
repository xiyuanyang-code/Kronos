import os
import pandas as pd
from datetime import datetime
import time
import shutil
from joblib import Parallel, delayed


def print_begin(begin):
    print(begin, end='', flush=True)


def print_end(end, begin):
    padding = ' ' * (len(begin) - len(end))
    print(f'\r{end}{padding}', flush=True)


class StockDownloader:
    def __init__(self, data_path, tushare_api):
        self.dir_name = os.path.join(data_path, 'tushare')
        self.tushare_api = tushare_api
        os.makedirs(self.dir_name, exist_ok=True)

    def get_data_by_date(self, date):
        """获取指定日期的股票数据"""
        dir_name = os.path.join(self.dir_name, date)
        os.makedirs(dir_name, exist_ok=True)
        for i in range(3):
            try:
                self.tushare_api.get_stock_by_date(date).to_csv(os.path.join(dir_name, 'stock.csv'), index=False)
                break
            except Exception:
                print(f'获取日线行情数据失败，第{i}次重试')
                time.sleep(2 ** i)
                continue
        else:
            # break未触发，抛出错误
            print('出错，请稍后重试')
            shutil.rmtree(dir_name)
            raise ValueError
        for i in range(3):
            try:
                self.tushare_api.get_factor_by_date(date).to_csv(os.path.join(dir_name, 'factor.csv'), index=False)
                break
            except Exception:
                time.sleep(2 ** i)
                continue
        else:
            # break未触发，抛出错误
            print('出错，请稍后重试')
            shutil.rmtree(dir_name)
            raise ValueError
        print(f"{date}数据下载完成")

    def update_by_trade_dates(self, start_date, end_date=None):
        """按交易日列表更新数据"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        trade_dates = self.tushare_api.get_trade_dates(start_date, end_date)
        exists = os.listdir(self.dir_name)
        for date in trade_dates:
            if date in exists:
                continue
            self.get_data_by_date(date)

    def export_daily_to_parquet(self, code_prefixes, start_date, end_date=None):
        """导出股票日线数据到Parquet格式"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        trade_dates = os.listdir(self.dir_name)
        trade_dates.sort()
        # 加载valid_date
        begin = '正在检查数据完整性...'
        print_begin(begin)
        valid_dates = []
        for date in trade_dates:
            if date < start_date or date > end_date:
                continue
            stock_file = os.path.join(self.dir_name, date, 'stock.csv')
            factor_file = os.path.join(self.dir_name, date, 'factor.csv')
            if not os.path.exists(stock_file) or not os.path.exists(factor_file):
                print(f'{date}数据缺失，请重新下载')
                shutil.rmtree(os.path.join(self.dir_name, date))
                raise FileNotFoundError
            else:
                valid_dates.append(date)
        print_end('数据完整性检查通过！', begin)
            
        def read_single_date(dir_name, date):
            stock_file = os.path.join(dir_name, date, 'stock.csv')
            factor_file = os.path.join(dir_name, date, 'factor.csv')
            df = pd.read_csv(stock_file)
            df = df[df['ts_code'].str.startswith(tuple(code_prefixes))]
            factor = pd.read_csv(factor_file)[['adj_factor', 'ts_code']]
            df = df.merge(factor, on='ts_code', how='left')
            return df
        
        begin = '正在读取数据...'
        print_begin(begin)
        dataframes = Parallel(n_jobs=-1)(
            delayed(read_single_date)(self.dir_name, date) for date in valid_dates
        )
        print_end('数据读取完成！', begin)
        
        print_begin('正在拼接数据...')
        df = pd.concat(dataframes, ignore_index=True)
        print_end('数据拼接完成！', begin)

        # 加工数据
        df.rename(columns={'ts_code': 'code', 'trade_date': 'day'}, inplace=True)  # 重命名列名
        df['code'] = df['code'].str.split('.').str[0].astype(int)
        return df
