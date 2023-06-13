from dataclasses import dataclass, field, make_dataclass
from typing import Type, Optional, Callable, Union, Iterable
from lamas.errors import ExpressionImplemError
from itertools import chain

NEG = "¬"

AND = "∧"
OR = "∨"

COND = "→"
BICOND = "↔"

TOP = "⊤"
BOT = "⊥"

TAgent = Union[str, int]


# Expr: Type

@dataclass
class Expr:
    """Base class for logical expressions.
    This should not be used directly, rather it exposes the generic
    operations beetween expressions:
    The basic ones have natural mappings to python operators
    - negate (~ Expr)
    - conjunct (Expr & Expr)
    - disjunct (Expr | Expr)
    For conditionals it's a bit of a stretch, but we have:
    - conditional implication (Expr >> Expr)
    - bi-conditional implication (Expr // Expr)

    The expression also has a length (the number of formal language definition applications
    required to build the formula)

    The expression indicates if it's modal and if it's dependent on an agent index, but these
    2 are special attributes fixed at the expression type level, so we never need to pass them
    when instantiating expressions. We foresee this attributes may be useful in further implementation phases

    """
    is_modal: bool = field(init=False, default=False)
    has_agent_idx: bool = field(init=False, default=False)

    def __invert__(self) -> "Expr":
        return Not(self)

    def __and__(self, other) -> "Expr":
        return And(self, other)

    def __or__(self, other) -> "Expr":
        return Or(self, other)

    def __rshift__(self, other) -> "Expr":
        return Cond(self, other)

    def __floordiv__(self, other) -> "Expr":
        return Bicond(self, other)

    def __len__(self):
        """Express the length of an expression:
        - atom = 1
        - unary op = 1 + len(operand)
        - binary op = 1 + len(left_operand) + len(right_operand)
        """
        raise ExpressionImplemError("Concrete expression types must define __len__")

    def atoms(self) -> Iterable["Atom"]:
        raise ExpressionImplemError("Concrete expression types must define `.atoms()`")


def mark_modal(cls):
    setattr(cls, "_is_modal", True)
    return cls


def set_ag_idx(cls):
    setattr(cls, "_has_agent_index", True)
    return cls


@dataclass
class Atom(Expr):
    """Expr type for an atomic proposition.
    This Simply wraps a string

    Examples:
         >>> Atom('p') # it is the case that p
         ...
         >>> Atom('q') # it is the case that q
         ...
    """
    symbol: str

    def __repr__(self):
        return self.symbol

    def __len__(self):
        return 1

    def atoms(self):
        return [self]


# --- Operator definers ----

def mk_op(base: Type, class_name: str, rator: Optional[str] = None):
    return make_dataclass(
        class_name,
        [('rator', str, field(default=rator or class_name, init=False))],
        bases=(base,),
        namespace={"__repr__": base.__repr__}
    )


@dataclass
class UnaryOp(Expr):
    """Base for simple unary operators.
    This wrap an ope-`rator` symbol string, and an ope-`rand` Expr.

    This should not be used directly, it is the "abstract" version rather.
    Use this to define actual unary operators via .define on the class itself

    Example:
        >>> Not = UnaryOp.define("Not", "¬")
        ...
        >>> Not(Atom('q')) # it is not the case that q
        ...
    """
    rand: Expr
    rator: str = field(default=None, init=False)

    def __repr__(self):
        return "{}{}".format(self.rator, repr(self.rand))

    @classmethod
    def define(cls, class_name: str, rator: Optional[str] = None) -> Callable[[Expr], "UnaryOp"]:
        """Define a unary operator given the class name and the rator symbol.
        If the rator sysmbol is not provided it defaults to the same string as the `class_name`
        """
        return mk_op(cls, class_name, rator)

    def __len__(self):
        return 1 + len(self.rand)

    def atoms(self):
        return self.rand.atoms()


@dataclass
class _Not(UnaryOp):
    rator: str = NEG


@dataclass
class AgentOperator(Expr):
    """Base for unary operators that are parametrized on an agent number/id.
    This wrap an ope-`rator` symbol string, and an ope-`rand` Expr plust the id of an agent

    This should not be used directly, it is the "abstract" version rather.
    Use this to define actual unary operators via .define on the class itself

    Example:
        >>> K = AgentOperator.define("K")
        ...
        >>> K(0,Atom('q')) # agent 0 knows that q
        ...
    """
    rator: str
    i: int
    rand: Expr
    is_modal: bool = field(init=False, default=True)  # agent operators are always modal
    has_agent_idx: bool = field(init=False, default=True)  # and index dependent

    def __repr__(self):
        return "{}_{} {}".format(self.rator, self.i, repr(self.rand))

    @classmethod
    def define(cls, class_name: str, rator: Optional[str] = None) -> Callable[[TAgent, Expr], "AgentOperator"]:
        return mk_op(cls, class_name, rator)  # type: ignore

    def __len__(self):
        return 1 + len(self.rand)

    def atoms(self):
        return self.rand.atoms()


@dataclass
class InfixBinOp(Expr):
    """Base for infix binary operators.
    This wrap an ope-`rator` symbol string, and 2 ope-`rand`s Expr (left `l_rand` and right `r_rand`)

    This should not be used directly, it is the "abstract" version rather.
    Use this to define actual unary operators via .define on the class itself

    Example:
        >>> And = InfixBinOp.define("And", "∧")
        ...
        >>> And(Atom('p'), ~Atom('q'))
        ...
    """
    rator: str
    l_rand: Expr
    r_rand: Expr

    def __repr__(self):
        return "({} {} {})".format(repr(self.l_rand), self.rator, repr(self.r_rand))

    @classmethod
    def define(cls, class_name: str, rator: Optional[str] = None) -> Callable[[Expr, Expr], "InfixBinOp"]:
        return mk_op(cls, class_name, rator)  # type: ignore

    def __len__(self):
        return 1 + len(self.l_rand) + len(self.r_rand)

    def atoms(self):
        return chain(self.l_rand.atoms(), self.r_rand.atoms())


# --- Operator Implementations ---


Not = UnaryOp.define("Not", NEG)
C = mark_modal(UnaryOp.define("C"))
E = mark_modal(UnaryOp.define("E"))

K = AgentOperator.define("K")
M = AgentOperator.define("M")

And = InfixBinOp.define("And", AND)
Or = InfixBinOp.define("Or", OR)
Cond = InfixBinOp.define("Cond", COND)
Bicond = InfixBinOp.define("Bicond", BICOND)

# the following is trick to obtain atoms for quick tests by just
# saying e.g. >>> P.q
# this always returns an atom for any dot notation, always creating a fresh atom
# with the part after the . as symbol. Do not use in practical scenarios, just for quick testing
P = type("__P__", (), {"__getattr__": lambda _, s: Atom(s)})()

if __name__ == "__main__":
    # Showcase the facility
    p = Atom('p')
    q = Atom('q')
    print(p)
    print(q)
    print(p.is_modal, p.has_agent_idx)  # False, False
    # combine manually
    print(Not(q))
    print(And(p, Not(q)))
    # combine using python operators
    print(~q)
    print(p & ~q)
    print((p & q) >> p)
    print(((p >> q) & p) >> q)
    # Epistemic operators with agent index
    print(K(0, q))
    print(K(0, q) & M(1, q))
    print(K(0, M(1, K(2, p | q))))

    print(K(0, q).is_modal)  # true
    print(K(0, q).has_agent_idx)  # true
    # Epistemic operators without agent index
    # - common knowledge
    print(C(p | q))
    print(~C(p & q))
    print(E(p | q))

    # len and atoms recursively
    large_formula = (P.a | P.b & ~P.c) >> (P.d // P.e)
    print(len(large_formula))
    print(list(large_formula.atoms()))
