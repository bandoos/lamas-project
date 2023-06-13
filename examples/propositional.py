"""
A simple example implementing formula checker for PL
using the expression facility implemented in lamas.expr.core
"""
from pprint import pprint

from lamas.expr.core import (
    Expr, Atom, Not, And, Or, Cond, Bicond, P
)
from lamas.errors import MissingValuation, UnsupportedExpressionType
from lamas.utils.dict_prod import dict_product
from typing import Dict, Iterable

Valuation = Dict[str, bool]


class StrictValuation(Valuation):
    """A simple valuation implemented via dict.
    All symbols used must be defined"""

    def __getitem__(self, item):
        if item in self:
            return self.get(item)
        raise MissingValuation(item, self)

    @classmethod
    def of(cls, trues, falses):
        """Quickly Create a valuation from a set of true and false symbols"""
        return cls({**{s: True for s in trues}, **{s: False for s in falses}})


def check(formula: Expr, valuation: StrictValuation):
    """Verify the truth value of `formula` under `valuation`
    by applying the truth defintions.
    TODO: this is a simple just-to-check, non-tail recursive implem, prone to stack-overflow!"""
    if isinstance(formula, Atom):
        return valuation[formula.symbol]
    elif isinstance(formula, Not):
        return not check(formula.rand, valuation)
    elif isinstance(formula, And):
        return check(formula.l_rand, valuation) and \
            check(formula.r_rand, valuation)
    elif isinstance(formula, Or):
        return check(formula.l_rand, valuation) or \
            check(formula.r_rand, valuation)
    elif isinstance(formula, Cond):
        return not check(formula.l_rand, valuation) or \
            check(formula.r_rand, valuation)
    elif isinstance(formula, Bicond):
        return check(Cond(formula.l_rand, formula.r_rand), valuation) and \
            check(Cond(formula.r_rand, formula.l_rand), valuation)
    else:
        raise UnsupportedExpressionType(f"PL checker does not support {type(formula)} expressions", expr=formula)


def gen_valuations(formula: Expr) -> Iterable[StrictValuation]:
    """Return a generator of StrictValuation that eventually covers
    all the truth table for `formula`'s atoms"""
    values = {atom.symbol: {True, False} for atom in formula.atoms()}
    return (StrictValuation(x) for x in dict_product(values))


def sat(formula: Expr):
    """Iterate all valuations for the atoms in `formula`, stop at the first one that satisfies
    the formula and return that valuation, else None
    """
    for val in gen_valuations(formula):
        if check(formula, val):
            return val
    return None


def check_taut(formula: Expr):
    """Iterate all valuations for the atoms in `formula`, stop at the first one that does not satisfy
    the formula and return False, the valuation/counter model. If all valuations sat the formula then it is a tautology,
    return True, None (no counter model)
    """
    for val in gen_valuations(formula):
        if not check(formula, val):
            return False, val
    return True, None


# --- put print logic around the functions defined above ---
def demo_check(f, v):
    is_satisfied = check(f, v)
    print("Formula `{}` is{} satisfied in {}".format(f, '' if is_satisfied else " NOT", v))


def demo_sat(f):
    example = sat(f)
    if example:
        print(f"Formula `{f}` is satisfied by valuation:")
        pprint(example)
    else:
        print(f"Formula `{f}` could NOT be satisfied by any valuation")


def demo_taut(f):
    verdict, counter_ex = check_taut(f)
    if verdict:
        print(f"The formula {f} IS a tautology")
    else:
        print(f"The formula {f} IS NOT a tautology\nCountermodel:")
        pprint(counter_ex)


if __name__ == "__main__":
    # showcase facility
    val_0 = StrictValuation.of(trues=["p"], falses=["q"])
    psi_0 = P.p | P.q
    demo_check(psi_0, val_0)
    print("-------")
    val_1 = StrictValuation.of(trues=["p", "q"], falses=["r"])
    psi_1 = (P.p & ~P.q) | P.r
    demo_check(psi_1, val_1)
    print("-------")
    demo_sat((P.p | P.q) & (P.r | P.s))
    print("-------")
    demo_sat(P.p & ~P.p)
    print("-------")
    demo_taut(P.p | ~P.p)
    demo_taut(~(P.p | P.q) // (~P.p & ~P.q))  # de-morgan 1
    demo_taut(~(P.p & P.q) // (~P.p | ~P.q))  # de-morgan 2
