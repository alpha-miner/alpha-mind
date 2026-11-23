from __future__ import annotations

from dataclasses import dataclass

import calc_flow as cf


def _expression(value):
    return (
        value
        if isinstance(value, Expression)
        else LAST(value)
        if isinstance(value, str)
        else Literal(value)
    )


def _groups(groups):
    if groups is None:
        return ()
    if isinstance(groups, str):
        return (groups,)
    return tuple(groups)


@dataclass(frozen=True, eq=False)
class Expression:
    op: str
    args: tuple
    label: str = None

    @property
    def fields(self):
        if self.op == "column":
            return [self.args[0]]
        dependencies = []
        for arg in self.args:
            if isinstance(arg, Expression):
                dependencies.extend(arg.fields)
        if self.op in ("cs_rank", "cs_quantile", "cs_res", "cs_top", "cs_bottom"):
            dependencies.extend(self.args[-1])
        return sorted(set(dependencies))

    @property
    def group_fields(self):
        fields = []
        if self.op in ("cs_rank", "cs_quantile", "cs_res", "cs_top", "cs_bottom"):
            fields.extend(self.args[-1])
        for arg in self.args:
            if isinstance(arg, Expression):
                fields.extend(arg.group_fields)
        return sorted(set(fields))

    @property
    def window(self):
        child_windows = [arg.window for arg in self.args if isinstance(arg, Expression)]
        if self.op == "shift":
            return child_windows[0] + self.args[1]
        if self.op in ("mean", "std", "sum", "max", "min", "ema"):
            return child_windows[0] + self.args[0] - 1
        if self.op == "macd":
            return child_windows[0] + self.args[1] - 1
        return max(child_windows, default=1)

    @property
    def requires_order(self):
        return (
            self.op.startswith("cs_")
            or self.op in ("shift", "mean", "std", "sum", "max", "min", "ema", "macd")
            or any(arg.requires_order for arg in self.args if isinstance(arg, Expression))
        )

    def bind(self, table):
        if self.op == "column":
            return table[self.args[0]]
        if self.op == "literal":
            return cf.lit(self.args[0])
        bound = [arg.bind(table) if isinstance(arg, Expression) else arg for arg in self.args]
        if self.op in ("add", "sub", "mul", "div", "pow", "neg"):
            if self.op == "neg":
                return -bound[0]
            left, right = bound
            if self.op == "add":
                return left + right
            if self.op == "sub":
                return left - right
            if self.op == "mul":
                return left * right
            if self.op == "div":
                return left / right
            return cf.row.pow(left, right)
        if self.op == "shift":
            return cf.ts.lag(bound[0], periods=self.args[1])
        if self.op in ("mean", "std", "sum", "max", "min", "ema"):
            value = bound[1]
            window = self.args[0]
            if self.op == "ema":
                return cf.ts.ema(value, span=window, min_periods=window)
            function = {
                "mean": cf.ts.mean,
                "std": cf.ts.stddev,
                "sum": cf.ts.sum,
                "max": cf.ts.max,
                "min": cf.ts.min,
            }[self.op]
            return function(value, window=cf.rows(window), min_periods=window)
        if self.op == "macd":
            return cf.ts.macd(
                bound[2], fast_span=self.args[0], slow_span=self.args[1], min_periods=self.args[1]
            )
        if self.op.startswith("cs_"):
            groups = self.args[-1]
            group = cf.exact_time(
                table["trade_date"], partition_by=tuple(table[name] for name in groups)
            )
            if self.op == "cs_rank":
                return cf.cs.rank(bound[0], group=group)
            if self.op == "cs_quantile":
                return cf.cs.percentile(bound[0], group=group)
            if self.op == "cs_res":
                return cf.cs.residual(bound[0], bound[1], group=group)
            if self.op == "cs_top":
                return cf.cs.top(bound[0], group=group, count=self.args[1])
            if self.op == "cs_bottom":
                return cf.cs.bottom(bound[0], group=group, count=self.args[1])
        raise ValueError(f"Unsupported expression operator: {self.op}")

    def __str__(self):
        if self.label:
            return self.label
        values = ", ".join(str(arg) for arg in self.args)
        return f"{self.op}({values})"

    def _binary(self, other, op, reflected=False):
        other = _expression(other)
        if other.op == "literal" and isinstance(other.args[0], int):
            other = Literal(float(other.args[0]))
        return Expression(op, (other, self) if reflected else (self, other))

    def __add__(self, other):
        return self._binary(other, "add")

    def __radd__(self, other):
        return self._binary(other, "add", True)

    def __sub__(self, other):
        return self._binary(other, "sub")

    def __rsub__(self, other):
        return self._binary(other, "sub", True)

    def __mul__(self, other):
        return self._binary(other, "mul")

    def __rmul__(self, other):
        return self._binary(other, "mul", True)

    def __truediv__(self, other):
        return self._binary(other, "div")

    def __rtruediv__(self, other):
        return self._binary(other, "div", True)

    def __pow__(self, other):
        return self._binary(other, "pow")

    def __xor__(self, other):
        return self._binary(other, "pow")

    def __neg__(self):
        return Expression("neg", (self,))


def Literal(value):
    return Expression("literal", (value,))


def LAST(x):
    return x if isinstance(x, Expression) else Expression("column", (x,))


def SHIFT(x, periods: int):
    if periods < 1:
        raise ValueError("periods must be positive")
    return Expression("shift", (_expression(x), int(periods)))


def CSRank(x, groups=None):
    return Expression("cs_rank", (_expression(x), _groups(groups)))


def CSQuantiles(x, groups=None):
    return Expression("cs_quantile", (_expression(x), _groups(groups)))


def CSRes(x, risk, groups=None):
    return Expression("cs_res", (_expression(x), _expression(risk), _groups(groups)))


def CSTopN(x, n: int, groups=None):
    return Expression("cs_top", (_expression(x), int(n), _groups(groups)))


def CSBottomN(x, n: int, groups=None):
    return Expression("cs_bottom", (_expression(x), int(n), _groups(groups)))


def _rolling(op, window, x):
    if isinstance(window, (Expression, str)):
        window, x = x, window
    if int(window) < 1:
        raise ValueError("window must be positive")
    return Expression(op, (int(window), _expression(x)))


def MA(window: int, x):
    return _rolling("mean", window, x)


def STD(window: int, x):
    return _rolling("std", window, x)


def SUM(window: int, x):
    return _rolling("sum", window, x)


def MAX(window: int, x):
    return _rolling("max", window, x)


def MIN(window: int, x):
    return _rolling("min", window, x)


def EMA(window: int, x):
    return _rolling("ema", window, x)


def MACD(x, short: int = 12, long: int = 26):
    if short >= long:
        raise ValueError("short must be less than long")
    return Expression("macd", (int(short), int(long), _expression(x)))
