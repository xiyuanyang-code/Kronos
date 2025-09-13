import pandas as pd
import numpy as np
import torch
from .factor.basic.stock_data import StockData, FeatureType
from .factor import FACTOR_MAP


class DataPreprocessor:
    """股票数据预处理工具类"""
    
    def apply_factor(self, df: pd.DataFrame, factor_name) -> pd.DataFrame:
        df = df.copy()
        df = self.convert_to_stock_data(df)
        df = FACTOR_MAP[factor_name.lower()]().process_data(df)
        return df

    @staticmethod
    def robust_norm(df: pd.DataFrame, fit_start: int, fit_end: int, filt=1) -> pd.DataFrame:
        print('正在进行 robust norm...')
        # 确保day列是整数类型
        df['day'] = df['day'].astype(int)
        # 不需要标准化的列
        exclude_cols = ['code', 'day', 'SecurityID', 'time']
        for col in list(df):
            if col.startswith('Ret'):
                exclude_cols.append(col)
        # 需要标准化的列
        numeric_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
        # 过滤掉numeric_cols中包含nan或inf的行
        if filt == 1:
            df.dropna(subset=numeric_cols, inplace=True)
        if filt == 2:
            df = df[~np.isinf(df).any(axis=1)]
        if filt == 3:
            df = df[~np.isinf(df).any(axis=1)]
            df.dropna(subset=numeric_cols, inplace=True)
        
        # 获取标准化区间数据
        fit_mask = (df['day'] >= fit_start) & (df['day'] <= fit_end)
        fit_df = df[fit_mask].copy()
        # 计算Robust标准化参数(中位数和四分位距)
        medians = fit_df[numeric_cols].median()
        q1 = fit_df[numeric_cols].quantile(0.25)
        q3 = fit_df[numeric_cols].quantile(0.75)
        iqrs = q3 - q1
        
        # 对所有数据进行Robust标准化
        for col in numeric_cols:
            df[col] = (df[col] - medians[col]) / (iqrs[col] + 1e-6)
        
        print('robust norm 完成')
        return df

    @staticmethod
    def convert_to_stock_data(df: pd.DataFrame, max_backtrack_days=260, max_future_days=0, features=None):
        # 获取特征
        if features is None:
            features = list(FeatureType)
        feature_cols = [f.name.lower() for f in features]

        # 创建完整的日期和股票代码组合
        all_days = sorted(df['day'].unique())
        all_codes = sorted(df['code'].unique())
        multi_index = pd.MultiIndex.from_product([all_days, all_codes], names=['day', 'code'])

        # 重新索引 DataFrame，填充缺失的日期和股票代码组合
        df = df.set_index(['day', 'code'])
        df = df.reindex(multi_index)

        df_unstacked = df.unstack(level='code')
        values = df_unstacked.values.reshape((len(all_days), len(feature_cols), len(all_codes)))

        preloaded_data = (torch.tensor(values, dtype=torch.float), all_days, all_codes)
        return StockData(
            instrument=all_codes,
            start_time=all_days[0],
            end_time=all_days[-1],
            max_backtrack_days=max_backtrack_days,
            max_future_days=max_future_days,
            features=features,
            preloaded_data=preloaded_data
        )
