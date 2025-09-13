from .base import *


class AI(FactorBase):
    name='ai'

    def get_expressions(self):
        ret_t11, ret_t11_std, ret_t11_rank_std = self.get_label(11)

        expressions = {
            'open': self.open,
            'close': self.close,
            'high': self.high,
            'low': self.low,
            'vol': self.volume,
            'amount': self.amount,
            'pre_close': self.pre_close,
            'change': self.change, 
            'pct_chg': self.pct_chg,
            'price_diff': self.high - self.low,
            'close_open_ratio': self.close / self.open,
            'volatility': (self.high - self.low) / self.pre_close,
            'amount_per_vol': self.amount / (self.volume + 1e-6),
            'vol_ma5': Mean(self.volume, 5),
            'vol_ma10': Mean(self.volume, 10),
            'close_ma5': Mean(self.close, 5),
            'close_ma10': Mean(self.close, 10),
            'momentum': self.close / Ref(self.close, 5) - 1,

            'Ret_t11': ret_t11,
            'Ret_t11_std': ret_t11_std,
            'Ret_t11_rank_std': ret_t11_rank_std,
        }
        return expressions
