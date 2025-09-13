from .basic.expression import *
from .basic.stock_data import FeatureType
import pandas as pd
from tqdm import tqdm


class FactorBase:
    def __init__(self):
        self.open = Feature(FeatureType.OPEN)
        self.high = Feature(FeatureType.HIGH)
        self.low = Feature(FeatureType.LOW)
        self.close = Feature(FeatureType.CLOSE)
        self.pre_close = Feature(FeatureType.PRE_CLOSE)
        self.change = Feature(FeatureType.CHANGE)
        self.pct_chg = Feature(FeatureType.PCT_CHG)
        self.volume = Feature(FeatureType.VOLUME)
        self.amount = Feature(FeatureType.AMOUNT)
        self.adj_factor = Feature(FeatureType.ADJ_FACTOR)
        self.vwap = self.amount / self.volume * 10
        self.apply_adj_factor()

    def apply_adj_factor(self):
        """对价格特征应用复权因子"""
        self.adj_open = self.open * self.adj_factor
        self.adj_high = self.high * self.adj_factor
        self.adj_low = self.low * self.adj_factor
        self.adj_close = self.close * self.adj_factor
        self.adj_vwap = self.vwap * self.adj_factor

    def get_expressions(self):
        """在此添加表达式，返回表达式的字典，键为特征名称，值为表达式"""
        ...

    def get_label(self, d=11, time='open'):
        """获取t+d预测的标签（复权）"""
        if time == 'open':
            price = self.open
        elif time == 'close':
            price = self.close
        else:
            raise NotImplementedError
        price = price * self.adj_factor
        price_t1 = Ref(price, -1)
        price_td = Ref(price, -d)
        ret_td = price_td / price_t1 - 1
        ret_td_std = CSNorm(ret_td)
        ret_td_rank_std = CSNorm(CSRank(ret_td))
        return ret_td, ret_td_std, ret_td_rank_std

    def process_data(self, raw_data):
        """将原始数据加工成需要的特征"""
        feature_expressions = self.get_expressions()
        feature_values = []
        for name, expr in tqdm(feature_expressions.items()):
            feature_values.append(expr.evaluate(raw_data))
        df = raw_data.make_dataframe(feature_values, list(feature_expressions.keys()))
        df.index.names = ['day', 'code']
        df.reset_index(inplace=True)
        # # 删除含有nan的行
        # df = df.dropna()
        return df
