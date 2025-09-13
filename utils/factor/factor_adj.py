from .base import *


class Adj(FactorBase):
    name='adj'

    def get_expressions(self):
        ret_t11, ret_t11_std, ret_t11_rank_std = self.get_label(11)

        expressions = {
            'open': self.adj_open,
            'close': self.adj_close,
            'high': self.adj_high,
            'low': self.adj_low,
            'vol': self.volume,
            'amount': self.amount,
            'adjvwap': self.adj_vwap,
            'Ret_t11': ret_t11,
            'Ret_t11_std': ret_t11_std,
            'Ret_t11_rank_std': ret_t11_rank_std,
        }
        return expressions
