from .base import *


class BS(FactorBase):
    name='bs'

    def get_expressions(self):
        ret_t11, ret_t11_std, ret_t11_rank_std = self.get_label(11)
        expressions = {
            # 基础价量因子
            'open': self.adj_open,
            'close': self.adj_close,
            'high': self.adj_high,
            'low': self.adj_low,
            'vol': self.volume,
            'amount': self.amount,
            # 特征
            'Ret_t11': ret_t11,
            'Ret_t11_std': ret_t11_std,
            'Ret_t11_rank_std': ret_t11_rank_std,
            }
        # 超买超卖因子
        momt = lambda t: (self.adj_close - Ref(self.adj_close, t)) / Ref(self.adj_close, t)
        expressions['mom1'] = momt(1)
        expressions['mom5'] = momt(5)
        expressions['mom10'] = momt(10)
        expressions['mom30'] = momt(30)
        expressions['mom5_sub_mom10'] = momt(5) - momt(10)
        expressions['mom5_sub_mom30'] = momt(5) - momt(30)
        expressions['close_div_ma5'] = self.adj_close / Mean(self.adj_close, 5)
        expressions['close_div_ma10'] = self.adj_close / Mean(self.adj_close, 10)
        expressions['close_div_ma30'] = self.adj_close / Mean(self.adj_close, 30)
        expressions['mom30_rank'] = CSRank(momt(30))
        expressions['high30_div_low30'] = Max(self.adj_high, 30) / Min(self.adj_low, 30)
        expressions['skew30'] = Skew(self.adj_close, 30)
        expressions['rvi'] = Mean(self.adj_close - self.adj_open, 30) / Mean(self.adj_high - self.adj_low, 30)
        expressions['vol5'] = Mean(self.volume, 5)
        expressions['vol10'] = Mean(self.volume, 10)
        expressions['vol30'] = Mean(self.volume, 30)
        expressions['vol_std5'] = Std(self.volume, 5)
        expressions['vol_std10'] = Std(self.volume, 10)
        expressions['vol_std30'] = Std(self.volume, 30)
        expressions['vol10_div_vol30'] = expressions['vol10'] / expressions['vol30']

        return expressions
