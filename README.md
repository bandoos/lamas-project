# lamas-project

Final Project Logical Aspects of Multi Agent Systems

## Status

This is currently not much complete, due to other course projects deadline and exams, we
have not managed to develop a lot as of now.

We have though started with essential pieces such as the construction and logical formulas and basic
model checking.

The module `lamas.expr.core` defines the implementation of recursive datatypes to represent
logical formulas including epistemic modal operators.

We are in the process of implementing actual Models/Structures but this is not done yet.

In the `examples` directory there are 2 simple examples to use the `Expr` data structures:

- propositional: implements a trivial brute-force SAT and Tautology checker for the subset of `Expr`
  types corresponding to Propositional logic.
    - `$ python -m examples.propositional`
- latex: implements a trivial recursive conversion from `Expr` to latex string.
    - `$ python -m examples.latex`

## Next steps

- [] Implement Kripke Model class `KStruct`
- [] Implement the various Axiom systems on to of `Expr` and `KStruct` with specific properties
- [] Implement basic validty checker for the modal systems
- [] Start implementing aspects specific to the project goal (Gossip)

# Dependencies

The project runs in python 3.10 but >3.6 should work.

The project currently has no dependencies.



