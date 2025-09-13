from .base import *


class Tecv2(FactorBase):
    name='tecv2'

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
        # DMI
        mtr = Sum(Greater(Greater(self.adj_high - self.adj_low, Abs(self.adj_high - Ref(self.adj_close, 1))), Abs(self.adj_low - Ref(self.adj_close, 1))), 14)
        hd = self.adj_high - Ref(self.adj_high, 1)
        ld = Ref(self.adj_low, 1) - self.adj_low
        dmp = Sum(GreaterThan(hd, 0) * GreaterThan(hd, ld) * hd, 14)
        dmm = Sum(GreaterThan(ld, 0) * GreaterThan(ld, hd) * ld, 14)
        expressions['pdi'] = dmp / mtr * 100
        expressions['mdi'] = dmm / mtr * 100
        expressions['adx'] = Mean(Abs(expressions['pdi'] - expressions['mdi']) / (expressions['pdi'] + expressions['mdi']) * 100, 6)
        expressions['adxr'] = (expressions['adx'] + Ref(expressions['adx'], 6)) / 2
        # CCI
        typ = (self.adj_high + self.adj_low + self.adj_close) / 3
        expressions['cci'] = (typ - Mean(typ, 14)) / (0.015 * Mean(Abs(typ - Mean(typ, 14)), 14))
        # BOLL
        std = Std(self.adj_close, 20)
        expressions['mb'] = Mean(self.adj_close, 20)
        expressions['up'] = expressions['mb'] + 2 * std
        expressions['dn'] = expressions['mb'] - 2 * std
        # BBI
        expressions['bbi'] = (Mean(self.adj_close, 3) + Mean(self.adj_close, 6) + Mean(self.adj_close, 12) + Mean(self.adj_close, 24)) / 4
        # MA
        expressions['ma5'] = Mean(self.adj_close, 5)
        expressions['ma10'] = Mean(self.adj_close, 10)
        expressions['ma20'] = Mean(self.adj_close, 20)
        expressions['ma30'] = Mean(self.adj_close, 30)
        expressions['ma60'] = Mean(self.adj_close, 60)
        expressions['ma120'] = Mean(self.adj_close, 120)
        # 成交量
        expressions['vol_ma5'] = Mean(self.volume, 5)
        expressions['vol_ma10'] = Mean(self.volume, 10)
        # # DDI
        # dmz = GreaterThan(self.adj_high + self.adj_low, Ref(self.adj_high + self.adj_low, 1)) * Greater(Abs(self.adj_high - Ref(self.adj_low, 1)), Abs(self.adj_low - Ref(self.adj_low, 1)))
        # dmf = LessThan(self.adj_high + self.adj_low, Ref(self.adj_high + self.adj_low, 1)) * Greater(Abs(self.adj_high - Ref(self.adj_low, 1)), Abs(self.adj_low - Ref(self.adj_low, 1)))
        # diz = Sum(dmz, 14) / (Sum(dmz, 14) + Sum(dmf, 14))
        # dif = Sum(dmf, 14) / (Sum(dmz, 14) + Sum(dmf, 14))
        # expressions['ddi'] = diz - dif
        # expressions['addi'] = Mean(expressions['ddi'], 6)

        return expressions
