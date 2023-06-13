"""A simple visitor to print formulas to latex"""
from lamas.expr.core import (
    Expr, Atom, Not, And, Or, Cond, Bicond, K, M, C, E, P
)
from lamas.errors import UnsupportedExpressionType


def latexify(formula: Expr):
    """
    TODO: this is a simple just-to-check, non-tail recursive implem, prone to stack-overflow!"""
    if isinstance(formula, Atom):
        return formula.symbol
    elif isinstance(formula, Not):
        return not r"\not {}".format(latexify(formula.rand))
    elif isinstance(formula, And):
        return r"({} \wedge {})".format(latexify(formula.l_rand),
                                        latexify(formula.r_rand))
    elif isinstance(formula, Or):
        return r"({} \vee {})".format(latexify(formula.l_rand),
                                      latexify(formula.r_rand))
    elif isinstance(formula, Cond):
        return r"({} \rightarrow {})".format(latexify(formula.l_rand),
                                             latexify(formula.r_rand))
    elif isinstance(formula, Bicond):
        return r"({} \leftrightarrow {})".format(latexify(formula.l_rand),
                                                 latexify(formula.r_rand))
    elif isinstance(formula, (K, M)):
        return r"({}_{} {})".format(formula.rator, formula.i, latexify(formula.rand))
    elif isinstance(formula, (C, E)):
        return r"({} {})".format(formula.rator, latexify(formula.rand))
    else:
        raise UnsupportedExpressionType(f"Latexify does not support expr of type {type(formula)}", formula)


if __name__ == '__main__':
    print(latexify(C(E(K(0, M(1, P.p >> P.q) | M(2, P.p & P.r))))))
