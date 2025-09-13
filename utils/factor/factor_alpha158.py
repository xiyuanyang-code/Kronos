from .base import *


class Alpha158(FactorBase):
    name='alpha158'

    def get_expressions(self):
        ret_t11, ret_t11_std, ret_t11_rank_std = self.get_label(11)
        expressions = {
            # 特征
            'Ret_t11': ret_t11,
            'Ret_t11_std': ret_t11_std,
            'Ret_t11_rank_std': ret_t11_rank_std,
            }
        
        open_ = self.adj_open
        close = self.adj_close
        high = self.adj_high
        low = self.adj_low
        volume = self.volume
        vwap = self.vwap
        # kbar
        expressions['KMID'] = (close - open_) / open_
        expressions['KLEN'] = (high - low) / open_
        expressions['KMID2'] = (close - open_) / (high - low + 1e-12)
        expressions['KUP'] = (high - Greater(open_, close)) / open_
        expressions['KUP2'] = (high - Greater(open_, close)) / (high - low + 1e-12)
        expressions['KLOW'] = (Less(open_, close) - low) / open_
        expressions['KLOW2'] = (Less(open_, close) - low) / (high - low + 1e-12)
        expressions['KSFT'] = (2 * close - high - low) / open_
        expressions['KSFT2'] = (2 * close - high - low) / (high - low + 1e-12)
        
        # price
        expressions[f'OPEN0'] = open_ / close
        expressions[f'HIGH0'] = high / close
        expressions[f'LOW0'] = low / close
        expressions[f'VWAP0'] = vwap / close
            
        # volume
        # for d in range(5):
        #     expressions[f'VOLUME{d}'] = Ref(volume, d) / (volume + 1e-12) if d != 0 else volume / (volume + 1e-12)
        
        # rolling
        for d in [5, 10, 20, 30, 60]:
            expressions[f'ROC{d}'] = Ref(close, d) / close
            expressions[f'MA{d}'] = Mean(close, d) / close
            expressions[f'STD{d}'] = Std(close, d) / close
            expressions[f'BETA{d}'] = Slope(close, d) / close
            expressions[f'RSQR{d}'] = Rsquare(close, d)
            expressions[f'RESI{d}'] = Resi(close, d) / close
            expressions[f'MAX{d}'] = Max(high, d) / close
            expressions[f'MIN{d}'] = Min(low, d) / close
            expressions[f'QTLU{d}'] = Quantile(close, d, 0.8) / close
            expressions[f'QTLD{d}'] = Quantile(close, d, 0.2) / close
            expressions[f'RANK{d}'] = Rank(close, d)
            expressions[f'RSV{d}'] = (close - Min(low, d)) / (Max(high, d) - Min(low, d) + 1e-12)
            expressions[f'IMAX{d}'] = IdxMax(high, d) / d
            expressions[f'IMIN{d}'] = IdxMin(low, d) / d
            expressions[f'IMXD{d}'] =  (IdxMax(high, d) - IdxMin(low, d)) / d
            expressions[f'CORR{d}'] = Corr(close, Log(volume + 1), d)
            expressions[f'CORD{d}'] = Corr(close / Ref(close, 1), Log(volume / Ref(volume, 1) + 1), d)
            expressions[f'CNTP{d}'] = Mean(GreaterThan(close, Ref(close, 1)), d)
            expressions[f'CNTN{d}'] = Mean(LessThan(close, Ref(close, 1)), d)
            expressions[f'CNTD{d}'] = Mean(GreaterThan(close, Ref(close, 1)), d) - Mean(LessThan(close, Ref(close, 1)), d)
            expressions[f'SUMP{d}'] = Sum(Greater(close - Ref(close, 1), 0), d) / (Sum(Abs(close - Ref(close, 1)), d) + 1e-12)
            expressions[f'SUMN{d}'] = Sum(Greater(Ref(close, 1) - close, 0), d) / (Sum(Abs(close - Ref(close, 1)), d) + 1e-12)
            expressions[f'SUMD{d}'] = (Sum(Greater(close - Ref(close, 1), 0), d) - Sum(Greater(Ref(close, 1) - close, 0), d)) / (Sum(Abs(close - Ref(close, 1)), d) + 1e-12)
            expressions[f'VMA{d}'] = Mean(volume, d) / (volume + 1e-12)
            expressions[f'VSTD{d}'] = Std(volume, d) / (volume + 1e-12)
            expressions[f'WVMA{d}'] = Std(Abs(close / Ref(close, 1) - 1) * volume, d) / (Mean(Abs(close / Ref(close, 1) - 1) * volume, d) + 1e-12)
            expressions[f'VSUMP{d}'] = Sum(Greater(volume - Ref(volume, 1), 0), d) / (Sum(Abs(volume - Ref(volume, 1)), d)+1e-12)
            expressions[f'VSUMN{d}'] = Sum(Greater(Ref(volume, 1) - volume, 0), d) / (Sum(Abs(volume - Ref(volume, 1)), d)+1e-12)
            expressions[f'VSUMD{d}'] = (Sum(Greater(volume - Ref(volume, 1), 0), d) - Sum(Greater(Ref(volume, 1) - volume, 0), d)) / (Sum(Abs(volume - Ref(volume, 1)), d) + 1e-12)
        return expressions
