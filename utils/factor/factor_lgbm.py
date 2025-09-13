from .base import *


class LGBM(FactorBase):
    name='lgbm'

    def get_expressions(self):
        ret_t11, ret_t11_std, ret_t11_rank_std = self.get_label(11)
        expressions = {
            # label
            'Ret_t11': ret_t11,
            'Ret_t11_std': ret_t11_std,
            'Ret_t11_rank_std': ret_t11_rank_std,
            }
        
        x = 20
        
        # 市价因子
        expressions['roc'] = (self.adj_close - Ref(self.adj_close, x)) / Ref(self.adj_close, x)
        expressions['ma'] = Mean(self.adj_close, x) / self.adj_close
        expressions['high'] = Max(self.adj_close, x) / self.adj_close
        expressions['low'] = Min(self.adj_close, x) / self.adj_close
        # 收盘价在过去x天的80%或20%分位数 / 今日收盘价
        # expressions['quan8']
        # expressions['quan2']
        
        # 动量因子
        up_mask = Sign(self.change) / 2 + 0.5
        down_mask = 0.5 - Sign(self.change) / 2
        expressions['numup'] = (Sum(up_mask, x) - Sum(down_mask, x)) / x
        expressions['sumup'] = Sum(up_mask * self.change, x) / Sum(Abs(self.change), x)
        expressions['vsumup'] = Sum(up_mask * self.volume, x) / Sum(self.volume, x)
        
        # 波动率与风险因子
        expressions['std'] = Std(self.adj_close, x) / self.adj_close
        expressions['vstd'] = Std(self.volume, x) / self.volume
        # x日收益率与成交量的加权标准差 / x日收益率与成交量的加权均值
        # expressions['wvma'] = WMA()
        
        # 市场深度和流动性因子
        # x日收盘价、高低价比率与对数成交量、日收益率与成交量变化率的相关系数
        expressions['corrpv'] = Corr(self.adj_close, Log(self.volume), x)
        expressions['corrhl'] = Corr(self.adj_high / self.adj_low, Log(self.volume), x)
        expressions['corrpvd'] = Corr(self.adj_close / Ref(self.adj_close, 1), self.volume / Ref(self.volume, 1), x)
        
        # 其他因子
        expressions['rsv'] = (self.adj_close - Min(self.adj_close, x)) / (Max(self.adj_close, x) - Min(self.adj_close, x))
        expressions['intra'] = Mean((self.adj_high - self.adj_low) / (self.adj_high + self.adj_low), x)
        expressions['td_open'] = self.adj_open / self.adj_close
        expressions['td_high'] = self.adj_high / self.adj_close
        expressions['td_low'] = self.adj_low / self.adj_close
        expressions['gt15'] = self.adj_open / Ref(self.adj_close, 1) - 1
        expressions['gt46'] = (Mean(self.adj_close, 3) + Mean(self.adj_close, 6) + Mean(self.adj_close, 12) + Mean(self.adj_close, 24)) / (4 * self.adj_close)
        # GT1 = 成交量变化的5日滚动排名 / 收益率的5日滚动排名
        # expressions['gt1'] = CSRank(Abs(self.volume / Ref(self.volume, 5))) / CSRank(self.adj_close / Ref(self.adj_close, 5))
        # GT2 = (今日收盘价-今日低点+今日高点-今日收盘价) / (今日高点-今日低点) 的差分值
        tmp = (self.adj_close - self.adj_low + self.adj_high - self.adj_close) / (self.adj_high - self.adj_low)
        expressions['gt2'] = tmp - Ref(tmp, 1)
        expressions['ret_5_30'] = self.adj_close / Ref(self.adj_close, 5) - self.adj_close / Ref(self.adj_close, 30)
        
        
        return expressions
