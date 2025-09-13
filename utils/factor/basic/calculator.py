from abc import ABCMeta, abstractmethod
from typing import Tuple, Optional, Sequence, Dict, Any
from torch import Tensor
import torch
import numpy as np  # For simulating data
import pandas as pd  # For simulating data

from utils.factor.basic.expression import *
from utils.factor.basic.stock_data import StockData

# !NOTE: Copied from alphagen
from utils.factor.basic.correlation import (
    normalize_by_day,
    batch_pearsonr,
    batch_spearmanr,
)

# *all the feature types are in utils.factor.basic.expression
# OPEN = 0
# HIGH = 1
# LOW = 2
# CLOSE = 3
# PRE_CLOSE = 4
# CHANGE = 5
# PCT_CHG = 6
# VOLUME = 7
# AMOUNT = 8
# ADJ_FACTOR = 9


# !我们在alphagen的Calculator的基础之上建立的新的自己的Calculator
# !NOTE: Copied from alphagen
class AlphaCalculator(metaclass=ABCMeta):
    @abstractmethod
    def calc_single_IC_ret(self, expr: Expression) -> float:
        "Calculate IC between a single alpha and a predefined target."

    @abstractmethod
    def calc_single_rIC_ret(self, expr: Expression) -> float:
        "Calculate Rank IC between a single alpha and a predefined target."

    def calc_single_all_ret(self, expr: Expression) -> Tuple[float, float]:
        return self.calc_single_IC_ret(expr), self.calc_single_rIC_ret(expr)

    @abstractmethod
    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        "Calculate IC between two alphas."

    @abstractmethod
    def calc_pool_IC_ret(
        self, exprs: Sequence[Expression], weights: Sequence[float]
    ) -> float:
        "First combine the alphas linearly,"
        "then Calculate IC between the linear combination and a predefined target."

    @abstractmethod
    def calc_pool_rIC_ret(
        self, exprs: Sequence[Expression], weights: Sequence[float]
    ) -> float:
        "First combine the alphas linearly,"
        "then Calculate Rank IC between the linear combination and a predefined target."

    @abstractmethod
    def calc_pool_all_ret(
        self, exprs: Sequence[Expression], weights: Sequence[float]
    ) -> Tuple[float, float]:
        "First combine the alphas linearly,"
        "then Calculate both IC and Rank IC between the linear combination and a predefined target."


# !NOTE: Copied from alphagen
class TensorAlphaCalculator(AlphaCalculator):
    def __init__(self, target: Optional[Tensor]) -> None:
        self._target = target

    @property
    @abstractmethod
    def n_days(self) -> int: ...

    @property
    def target(self) -> Tensor:
        if self._target is None:
            raise ValueError("A target must be set before calculating non-mutual IC.")
        return self._target

    @abstractmethod
    def evaluate_alpha(self, expr: Expression) -> Tensor:
        "Evaluate an alpha into a `Tensor` of shape (days, stocks)."

    def make_ensemble_alpha(
        self, exprs: Sequence[Expression], weights: Sequence[float]
    ) -> Tensor:
        n = len(exprs)
        factors = [self.evaluate_alpha(exprs[i]) * weights[i] for i in range(n)]
        # Stack factors along a new dimension (dim=0), then sum along that dimension
        return torch.sum(torch.stack(factors, dim=0), dim=0)

    def _calc_IC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_pearsonr(value1, value2).mean().item()

    def _calc_rIC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_spearmanr(value1, value2).mean().item()

    def _IR_from_batch(self, batch: Tensor) -> float:
        mean, std = batch.mean(), batch.std()
        return (mean / std).item() if std != 0 else 0.0  # Avoid division by zero

    def _calc_ICIR(self, value1: Tensor, value2: Tensor) -> float:
        return self._IR_from_batch(batch_pearsonr(value1, value2))

    def _calc_rICIR(self, value1: Tensor, value2: Tensor) -> float:
        return self._IR_from_batch(batch_spearmanr(value1, value2))

    def calc_single_IC_ret(self, expr: Expression) -> float:
        return self._calc_IC(self.evaluate_alpha(expr), self.target)

    def calc_single_IC_ret_daily(self, expr: Expression) -> Tensor:
        return batch_pearsonr(self.evaluate_alpha(expr), self.target)

    def calc_single_rIC_ret(self, expr: Expression) -> float:
        return self._calc_rIC(self.evaluate_alpha(expr), self.target)

    def calc_single_all_ret(self, expr: Expression) -> Tuple[float, float]:
        value = self.evaluate_alpha(expr)
        target = self.target
        return self._calc_IC(value, target), self._calc_rIC(value, target)

    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        return self._calc_IC(self.evaluate_alpha(expr1), self.evaluate_alpha(expr2))

    def calc_mutual_IC_daily(self, expr1: Expression, expr2: Expression) -> Tensor:
        return batch_pearsonr(self.evaluate_alpha(expr1), self.evaluate_alpha(expr2))

    def calc_pool_IC_ret(
        self, exprs: Sequence[Expression], weights: Sequence[float]
    ) -> float:
        with torch.no_grad():
            value = self.make_ensemble_alpha(exprs, weights)
            return self._calc_IC(value, self.target)

    def calc_pool_rIC_ret(
        self, exprs: Sequence[Expression], weights: Sequence[float]
    ) -> float:
        with torch.no_grad():
            value = self.make_ensemble_alpha(exprs, weights)
            return self._calc_rIC(value, self.target)

    def calc_pool_all_ret(
        self, exprs: Sequence[Expression], weights: Sequence[float]
    ) -> Tuple[float, float]:
        with torch.no_grad():
            value = self.make_ensemble_alpha(exprs, weights)
            target = self.target
            return self._calc_IC(value, target), self._calc_rIC(value, target)

    def calc_pool_all_ret_with_ir(
        self, exprs: Sequence[Expression], weights: Sequence[float]
    ) -> Tuple[float, float, float, float]:
        "Returns IC, ICIR, Rank IC, Rank ICIR"
        with torch.no_grad():
            value = self.make_ensemble_alpha(exprs, weights)
            target = self.target
            ics = batch_pearsonr(value, target)
            rics = batch_spearmanr(value, target)
            ic_mean, ic_std = ics.mean().item(), ics.std().item()
            ric_mean, ric_std = rics.mean().item(), rics.std().item()
            # Handle potential division by zero for IR
            icir = ic_mean / ic_std if ic_std != 0 else 0.0
            ricir = ric_mean / ric_std if ric_std != 0 else 0.0
            return ic_mean, icir, ric_mean, ricir


class FinalAlphaCalculator(TensorAlphaCalculator):
    def __init__(self, data: StockData, target: Optional[Expression] = None):
        super().__init__(
            normalize_by_day(target.evaluate(data)) if target is not None else None
        )
        self.data = data

    def evaluate_alpha(self, expr: Expression) -> Tensor:
        return normalize_by_day(expr.evaluate(self.data))

    @property
    def n_days(self) -> int:
        return self.data.n_days
