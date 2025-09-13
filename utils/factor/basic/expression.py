from abc import ABCMeta, abstractmethod
from typing import List, Type, Union, Tuple

import torch
from torch import Tensor
from .maybe import Maybe, some, none
from .stock_data import StockData, FeatureType


_ExprOrFloat = Union["Expression", float]
_DTimeOrInt = Union["DeltaTime", int]


class OutOfDataRangeError(IndexError):
    pass


class Expression(metaclass=ABCMeta):
    @abstractmethod
    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor: ...

    def __repr__(self) -> str: return str(self)

    def __add__(self, other: _ExprOrFloat) -> "Add": return Add(self, other)
    def __radd__(self, other: float) -> "Add": return Add(other, self)
    def __sub__(self, other: _ExprOrFloat) -> "Sub": return Sub(self, other)
    def __rsub__(self, other: float) -> "Sub": return Sub(other, self)
    def __mul__(self, other: _ExprOrFloat) -> "Mul": return Mul(self, other)
    def __rmul__(self, other: float) -> "Mul": return Mul(other, self)
    def __truediv__(self, other: _ExprOrFloat) -> "Div": return Div(self, other)
    def __rtruediv__(self, other: float) -> "Div": return Div(other, self)
    def __pow__(self, other: _ExprOrFloat) -> "Pow": return Pow(self, other)
    def __rpow__(self, other: float) -> "Pow": return Pow(other, self)
    def __pos__(self) -> "Expression": return self
    def __neg__(self) -> "Sub": return Sub(0., self)
    def __abs__(self) -> "Abs": return Abs(self)

    @property
    @abstractmethod
    def is_featured(self) -> bool: ...


class Feature(Expression):
    def __init__(self, feature: FeatureType) -> None:
        self._feature = feature

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        assert period.step == 1 or period.step is None
        if period.start < -data.max_backtrack_days:
            raise OutOfDataRangeError()
        elif period.stop - 1 > data.max_future_days:  # 对于超出范围的未来标签，使用nan填充，避免最新数据无法预测
            start = period.start + data.max_backtrack_days
            stop = data.max_backtrack_days + data.max_future_days + data.n_days
            valid_result = data.data[start:stop, int(self._feature), :]
            nan_part = torch.full((period.stop - 1 - data.max_future_days, data.n_stocks), torch.nan)
            return torch.cat([valid_result, nan_part], dim=0)
        else:
            start = period.start + data.max_backtrack_days
            stop = period.stop + data.max_backtrack_days + data.n_days - 1
            return data.data[start:stop, int(self._feature), :]

    def __str__(self) -> str: return '$' + self._feature.name.lower()

    @property
    def is_featured(self): return True


class Constant(Expression):
    def __init__(self, value: float) -> None:
        self.value = value

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        assert period.step == 1 or period.step is None
        if (period.start < -data.max_backtrack_days or
                period.stop - 1 > data.max_future_days):
            raise OutOfDataRangeError()
        device = data.data.device
        dtype = data.data.dtype
        days = period.stop - period.start - 1 + data.n_days
        return torch.full(size=(days, data.n_stocks),
                          fill_value=self.value, dtype=dtype, device=device)

    def __str__(self) -> str: return str(self.value)

    @property
    def is_featured(self): return False


class DeltaTime(Expression):
    # This is not something that should be in the final expression
    # It is only here for simplicity in the implementation of the tree builder
    def __init__(self, delta_time: int) -> None:
        self._delta_time = delta_time

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        assert False, "Should not call evaluate on delta time"

    def __str__(self) -> str: return f"{self._delta_time}d"

    @property
    def is_featured(self): return False


def _into_expr(value: _ExprOrFloat) -> "Expression":
    return value if isinstance(value, Expression) else Constant(value)


def _into_delta_time(value: Union[int, DeltaTime]) -> DeltaTime:
    return value if isinstance(value, DeltaTime) else DeltaTime(value)


# Operator base classes

class Operator(Expression):
    @classmethod
    @abstractmethod
    def n_args(cls) -> int: ...

    @classmethod
    @abstractmethod
    def category_type(cls) -> Type["Operator"]: ...

    @classmethod
    @abstractmethod
    def validate_parameters(cls, *args) -> Maybe[str]: ...

    @classmethod
    def _check_arity(cls, *args) -> Maybe[str]:
        arity = cls.n_args()
        if len(args) == arity:
            return none(str)
        else:
            return some(f"{cls.__name__} expects {arity} operand(s), but received {len(args)}")

    @classmethod
    def _check_exprs_featured(cls, args: list) -> Maybe[str]:
        any_is_featured: bool = False
        for i, arg in enumerate(args):
            if not isinstance(arg, (Expression, float)):
                return some(f"{arg} is not a valid expression")
            if isinstance(arg, DeltaTime):
                return some(f"{cls.__name__} expects a normal expression for operand {i + 1}, "
                            f"but got {arg} (a DeltaTime)")
            any_is_featured = any_is_featured or (isinstance(arg, Expression) and arg.is_featured)
        if not any_is_featured:
            if len(args) == 1:
                return some(f"{cls.__name__} expects a featured expression for its operand, "
                            f"but {args[0]} is not featured")
            else:
                return some(f"{cls.__name__} expects at least one featured expression for its operands, "
                            f"but none of {args} is featured")
        return none(str)

    @classmethod
    def _check_delta_time(cls, arg) -> Maybe[str]:
        if not isinstance(arg, (DeltaTime, int)):
            return some(f"{cls.__name__} expects a DeltaTime as its last operand, but {arg} is not")
        return none(str)

    @property
    @abstractmethod
    def operands(self) -> Tuple[Expression, ...]: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({','.join(str(op) for op in self.operands)})"


class UnaryOperator(Operator):
    def __init__(self, operand: _ExprOrFloat) -> None:
        self._operand = _into_expr(operand)

    @classmethod
    def n_args(cls) -> int: return 1

    @classmethod
    def category_type(cls): return UnaryOperator

    @classmethod
    def validate_parameters(cls, *args) -> Maybe[str]:
        return cls._check_arity(*args).or_else(lambda: cls._check_exprs_featured([args[0]]))

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        return self._apply(self._operand.evaluate(data, period))

    @abstractmethod
    def _apply(self, operand: Tensor) -> Tensor: ...

    @property
    def operands(self): return self._operand,

    @property
    def is_featured(self): return self._operand.is_featured


class BinaryOperator(Operator):
    def __init__(self, lhs: _ExprOrFloat, rhs: _ExprOrFloat) -> None:
        self._lhs = _into_expr(lhs)
        self._rhs = _into_expr(rhs)

    @classmethod
    def n_args(cls) -> int: return 2

    @classmethod
    def category_type(cls): return BinaryOperator

    @classmethod
    def validate_parameters(cls, *args) -> Maybe[str]:
        return cls._check_arity(*args).or_else(lambda: cls._check_exprs_featured([args[0], args[1]]))

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        return self._apply(self._lhs.evaluate(data, period), self._rhs.evaluate(data, period))

    @abstractmethod
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: ...

    def __str__(self) -> str: return f"{type(self).__name__}({self._lhs},{self._rhs})"

    @property
    def operands(self): return self._lhs, self._rhs

    @property
    def is_featured(self): return self._lhs.is_featured or self._rhs.is_featured


class RollingOperator(Operator):
    def __init__(self, operand: _ExprOrFloat, delta_time: _DTimeOrInt) -> None:
        self._operand = _into_expr(operand)
        self._delta_time = _into_delta_time(delta_time)._delta_time

    @classmethod
    def n_args(cls) -> int: return 2

    @classmethod
    def category_type(cls): return RollingOperator

    @classmethod
    def validate_parameters(cls, *args) -> Maybe[str]:
        return cls._check_arity(*args).or_else(
            lambda: cls._check_exprs_featured([args[0]])
        ).or_else(
            lambda: cls._check_delta_time(args[1])
        )

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time + 1
        stop = period.stop
        # L: period length (requested time window length)
        # W: window length (dt for rolling)
        # S: stock count
        values = self._operand.evaluate(data, slice(start, stop))   # (L+W-1, S)
        values = values.unfold(0, self._delta_time, 1)              # (L, S, W)
        return self._apply(values)                                  # (L, S)

    @abstractmethod
    def _apply(self, operand: Tensor) -> Tensor: ...

    @property
    def operands(self): return self._operand, DeltaTime(self._delta_time)

    @property
    def is_featured(self): return self._operand.is_featured


class PairRollingOperator(Operator):
    def __init__(self, lhs: _ExprOrFloat, rhs: _ExprOrFloat, delta_time: _DTimeOrInt) -> None:
        self._lhs = _into_expr(lhs)
        self._rhs = _into_expr(rhs)
        self._delta_time = _into_delta_time(delta_time)._delta_time

    @classmethod
    def n_args(cls) -> int: return 3

    @classmethod
    def category_type(cls): return PairRollingOperator

    @classmethod
    def validate_parameters(cls, *args) -> Maybe[str]:
        return cls._check_arity(*args).or_else(
            lambda: cls._check_exprs_featured([args[0], args[1]])
        ).or_else(
            lambda: cls._check_delta_time(args[2])
        )

    def _unfold_one(self, expr: Expression,
                    data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time + 1
        stop = period.stop
        # L: period length (requested time window length)
        # W: window length (dt for rolling)
        # S: stock count
        values = expr.evaluate(data, slice(start, stop))            # (L+W-1, S)
        return values.unfold(0, self._delta_time, 1)                # (L, S, W)

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        lhs = self._unfold_one(self._lhs, data, period)
        rhs = self._unfold_one(self._rhs, data, period)
        return self._apply(lhs, rhs)                                # (L, S)

    @abstractmethod
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: ...

    @property
    def operands(self): return self._lhs, self._rhs, DeltaTime(self._delta_time)

    @property
    def is_featured(self): return self._lhs.is_featured or self._rhs.is_featured


# # 三元表达式
# class TeriaryExpression(Operator):
#     def __init__(self, condition: _ExprOrFloat, true_value: _ExprOrFloat, false_value: _ExprOrFloat) -> None:
#         self._condition = _into_expr(condition)
#         self._true_value = _into_expr(true_value)
#         self._false_value = _into_expr(false_value)
    
#     @classmethod
#     def n_args(cls) -> int: return 3

#     @classmethod
#     def category_type(cls): return TeriaryExpression

#     @classmethod
#     def validate_parameters(cls, *args) -> Maybe[str]:
#         return cls._check_arity(*args).or_else(
#             lambda: cls._check_exprs_featured([args[0], args[1], args[2]])
#         )

#     def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
#         condition = self._condition.evaluate(data, period)
#         true_value = self._true_value.evaluate(data, period)
#         false_value = self._false_value.evaluate(data, period)
#         print(condition)
#         return torch.where(condition > 0, true_value, false_value)
    
#     @property
#     def operands(self): return self._condition, self._true_value, self._false_value

#     @property
#     def is_featured(self): return self._condition.is_featured or self._true_value.is_featured or self._false_value.is_featured


# Operator implementations

class Abs(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.abs()


class Sign(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.sign()


class Log(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.log()


class CSRank(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        nan_mask = operand.isnan()
        n = (~nan_mask).sum(dim=1, keepdim=True)
        rank = operand.argsort().argsort() / n
        rank[nan_mask] = torch.nan
        return rank


class CSNorm(UnaryOperator):  # 截面标准化
    def _apply(self, operand: Tensor) -> Tensor:
        # operand: (day, code)
        mean = operand.nanmean(dim=1, keepdim=True)
        std = ((operand - mean) ** 2).nanmean(dim=1, keepdim=True).sqrt()
        eps = 1e-6
        std[std < eps] = eps  # Avoid division by zero
        normalized = (operand - mean) / std
        return normalized


class Add(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs + rhs


class Sub(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs - rhs


class Mul(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs * rhs


class Div(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs / rhs


class Pow(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs ** rhs


class Greater(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs.max(rhs)


class Less(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs.min(rhs)


class GreaterThan(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return (lhs > rhs).float()

class LessThan(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return (lhs < rhs).float()


class Ref(RollingOperator):
    # Ref is not *really* a rolling operator, in that other rolling operators
    # deal with the values in (-dt, 0], while Ref only deal with the values
    # at -dt. Nonetheless, it should be classified as rolling since it modifies
    # the time window.

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time
        stop = period.stop - self._delta_time
        return self._operand.evaluate(data, slice(start, stop))

    def _apply(self, operand: Tensor) -> Tensor:
        # This is just for fulfilling the RollingOperator interface
        ...


class Mean(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.mean(dim=-1)


class Sum(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.sum(dim=-1)


class Std(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.std(dim=-1)


class Var(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.var(dim=-1)


class Skew(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        # skew = m3 / m2^(3/2)
        central = operand - operand.mean(dim=-1, keepdim=True)
        m3 = (central ** 3).mean(dim=-1)
        m2 = (central ** 2).mean(dim=-1)
        return m3 / m2 ** 1.5


class Kurt(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        # kurt = m4 / var^2 - 3
        central = operand - operand.mean(dim=-1, keepdim=True)
        m4 = (central ** 4).mean(dim=-1)
        var = operand.var(dim=-1)
        return m4 / var ** 2 - 3


class Max(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.max(dim=-1)[0]


class Min(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.min(dim=-1)[0]


class Med(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.median(dim=-1)[0]


class Mad(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        central = operand - operand.mean(dim=-1, keepdim=True)
        return central.abs().mean(dim=-1)


class Rank(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        last = operand[:, :, -1, None]
        left = (last < operand).count_nonzero(dim=-1)
        right = (last <= operand).count_nonzero(dim=-1)
        result = (right + left + (right > left)) / (2 * n)
        return result


class Delta(RollingOperator):
    # Delta is not *really* a rolling operator, in that other rolling operators
    # deal with the values in (-dt, 0], while Delta only deal with the values
    # at -dt and 0. Nonetheless, it should be classified as rolling since it
    # modifies the time window.

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time
        stop = period.stop
        values = self._operand.evaluate(data, slice(start, stop))
        return values[self._delta_time:] - values[:-self._delta_time]

    def _apply(self, operand: Tensor) -> Tensor:
        # This is just for fulfilling the RollingOperator interface
        ...


class WMA(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        weights = torch.arange(n, dtype=operand.dtype, device=operand.device)
        weights /= weights.sum()
        return (weights * operand).sum(dim=-1)


class EMA(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        alpha = 1 - 2 / (1 + n)
        power = torch.arange(n, 0, -1, dtype=operand.dtype, device=operand.device)
        weights = alpha ** power
        weights /= weights.sum()
        return (weights * operand).sum(dim=-1)


class Slope(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        # 计算过去 d 天的价格变化
        price_change = operand[:, :, -1] - operand[:, :, 0]
        # 获取最新收盘价
        latest_close_price = operand[:, :, -1]
        # 避免除零错误
        eps = 1e-6
        latest_close_price[latest_close_price < eps] = eps
        # 计算斜率
        slope = price_change / latest_close_price
        return slope
    

class Rsquare(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        # 天数
        n = operand.shape[-1]
        # 时间序列，从 1 到 n
        time = torch.arange(1, n + 1, dtype=operand.dtype, device=operand.device)
        # 计算均值
        y_mean = operand.mean(dim=-1, keepdim=True)
        # 计算回归系数
        numerator = (n * (operand * time).sum(dim=-1) - operand.sum(dim=-1) * time.sum())
        denominator = (n * (time ** 2).sum() - time.sum() ** 2)
        beta = numerator / denominator
        alpha = y_mean.squeeze(-1) - beta * (time.mean())
        # 预测值
        y_pred = alpha.unsqueeze(-1) + beta.unsqueeze(-1) * time
        # 总平方和
        ss_total = ((operand - y_mean) ** 2).sum(dim=-1)
        # 残差平方和
        ss_residual = ((operand - y_pred) ** 2).sum(dim=-1)
        # 计算 R^2
        r_square = 1 - ss_residual / (ss_total + 1e-12)
        return r_square


class Resi(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        # 天数
        n = operand.shape[-1]
        # 时间序列，从 1 到 n
        time = torch.arange(1, n + 1, dtype=operand.dtype, device=operand.device)
        # 计算均值
        y_mean = operand.mean(dim=-1, keepdim=True)
        # 计算回归系数
        numerator = (n * (operand * time).sum(dim=-1) - operand.sum(dim=-1) * time.sum())
        denominator = (n * (time ** 2).sum() - time.sum() ** 2)
        beta = numerator / denominator
        alpha = y_mean.squeeze(-1) - beta * (time.mean())
        # 预测值
        y_pred = alpha.unsqueeze(-1) + beta.unsqueeze(-1) * time
        # 计算残差
        residuals = operand - y_pred
        # 取最后一天的残差值作为结果
        return residuals[:, :, -1]
    

class IdxMax(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        # 计算每个滚动窗口内最大值的索引
        max_indices = operand.argmax(dim=-1)
        # 窗口大小
        window_size = operand.shape[-1]
        # 计算当前日期与最大值日期之间的天数
        days_since_max = window_size - 1 - max_indices
        return days_since_max
    

class IdxMin(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        # 计算每个滚动窗口内最小值的索引
        min_indices = operand.argmin(dim=-1)
        # 窗口大小
        window_size = operand.shape[-1]
        # 计算当前日期与最小值日期之间的天数
        days_since_min = window_size - 1 - min_indices
        return days_since_min


class Quantile(RollingOperator):
    def __init__(self, operand: _ExprOrFloat, delta_time: _DTimeOrInt, quantile: float) -> None:
        super().__init__(operand, delta_time)
        self._quantile = quantile

    @classmethod
    def validate_parameters(cls, *args) -> Maybe[str]:
        check_arity = cls._check_arity(*args)
        if check_arity.is_some():
            return check_arity

        check_exprs_featured = cls._check_exprs_featured([args[0]])
        if check_exprs_featured.is_some():
            return check_exprs_featured

        check_delta_time = cls._check_delta_time(args[1])
        if check_delta_time.is_some():
            return check_delta_time

        if not isinstance(args[2], float) or args[2] < 0 or args[2] > 1:
            return some(f"{cls.__name__} expects a valid quantile value (between 0 and 1) for its third operand, but {args[2]} is not valid.")

        return none(str)

    def _apply(self, operand: Tensor) -> Tensor:
        # 计算指定分位数
        quantile_values = torch.quantile(operand, self._quantile, dim=-1)
        # 获取最新收盘价
        latest_close_price = operand[:, :, -1]
        # 避免除零错误
        eps = 1e-6
        latest_close_price[latest_close_price < eps] = eps
        # 计算结果
        result = quantile_values / latest_close_price
        return result


class Cov(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        n = lhs.shape[-1]
        clhs = lhs - lhs.mean(dim=-1, keepdim=True)
        crhs = rhs - rhs.mean(dim=-1, keepdim=True)
        return (clhs * crhs).sum(dim=-1) / (n - 1)


class Corr(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        clhs = lhs - lhs.mean(dim=-1, keepdim=True)
        crhs = rhs - rhs.mean(dim=-1, keepdim=True)
        ncov = (clhs * crhs).sum(dim=-1)
        nlvar = (clhs ** 2).sum(dim=-1)
        nrvar = (crhs ** 2).sum(dim=-1)
        stdmul = (nlvar * nrvar).sqrt()
        stdmul[(nlvar < 1e-6) | (nrvar < 1e-6)] = 1
        return ncov / stdmul


Operators: List[Type[Operator]] = [
    # Unary
    Abs, Sign, Log, CSRank,
    # Binary
    Add, Sub, Mul, Div, Pow, Greater, Less,
    # Rolling
    Ref, Mean, Sum, Std, Var, Skew, Kurt, Max, Min,
    Med, Mad, Rank, Delta, WMA, EMA,
    # Pair rolling
    Cov, Corr
]

OPERATORS = Operators