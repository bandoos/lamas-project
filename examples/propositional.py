"""
A simple example implementing formula checker for PL
using the expression facility implemented in lamas.expr.core
"""
from lamas.expr.core import (
    Expr, Atom, Not, And, Or, Cond, Bicond, P
)
from lamas.errors import MissingValuation, UnsupportedExpressionType
from dataclasses import dataclass
from typing import Dict
from functools import singledispatchmethod

Valuation = Dict[str, bool]


class StrictValuation(Valuation):
    def __getitem__(self, item):
        if item in self:
            return self.get(item)
        raise MissingValuation(item, self)

    @classmethod
    def of(cls, trues, falses):
        return cls({**{s: True for s in trues}, **{s: False for s in falses}})


def check(formula: Expr, valuation: StrictValuation):
    if isinstance(formula, Atom):
        return valuation[formula.symbol]
    elif isinstance(formula, Not):
        return not check(formula.rand, valuation)
    elif isinstance(formula, And):
        return check(formula.l_rand, valuation) and check(formula.r_rand, valuation)
    elif isinstance(formula, Or):
        return check(formula.l_rand, valuation) or check(formula.r_rand, valuation)
    elif isinstance(formula, Cond):
        return not check(formula.l_rand, valuation) or check(formula.r_rand, valuation)
    elif isinstance(formula, Bicond):
        return check(Cond(formula.l_rang, formula.r_rand), valuation) and check(Cond(formula.r_rang, formula.l_rand),
                                                                                valuation)
    else:
        raise UnsupportedExpressionType(f"PL checker does not support {type(formula)} expressions", expr=formula)


if __name__ == "__main__":
    def demo(f, v):
        is_satisfied = check(f, v)
        print("Formula {} is{} satisfied in {}".format(f, '' if is_satisfied else " NOT", v))


    ...
    val_0 = StrictValuation.of(trues=["p"], falses=["q"])
    psi_0 = P.p | P.q
    demo(psi_0, val_0)

    val_1 = StrictValuation.of(trues=["p", "q"], falses=["r"])
    psi_1 = (P.p & ~P.q) | P.r
    demo(psi_1, val_1)
