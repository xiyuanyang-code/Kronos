from .base import *


class Tec(FactorBase):
    name='tec'

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
        # 技术指标
        # 量比
        expressions['qrr5'] = self.volume / Mean(self.volume, 5)
        expressions['qrr10'] = self.volume / Mean(self.volume, 10)
        # MACD
        diff = EMA(self.adj_close, 12) - EMA(self.adj_close, 26)
        dea = EMA(diff, 9)
        macd = 2 * (diff - dea)
        expressions['macd'] = macd
        expressions['macd_diff'] = diff
        expressions['macd_dea'] = dea
        # KDJ
        rsv = (self.adj_close - Min(self.adj_low, 9)) / (Max(self.adj_high, 9) - Min(self.adj_low, 9)) * 100
        k = EMA(rsv, 3)
        d = EMA(k, 3)
        j = 3 * k - 2 * d
        expressions['k'] = k
        expressions['d'] = d
        expressions['j'] = j
        # RSI
        change_pos = Greater(self.adj_close - Ref(self.adj_close, 1), 0)  # 收盘价变化值的正部
        change_neg = Greater(Ref(self.adj_close, 1) - self.adj_close, 0)  # 负部
        rst = lambda t: Mean(change_pos, t) / Mean(change_neg, t)
        expressions['rsi6'] = 100 - 100 / (1 + rst(6))
        expressions['rsi12'] = 100 - 100 / (1 + rst(12))
        expressions['rsi24'] = 100 - 100 / (1 + rst(24))
        return expressions
