from itertools import product
from collections.abc import Iterable
from typing import List, Dict


def ensure_iterable(x):
    if isinstance(x, Iterable):
        return x
    else:
        return [x]


def merge_dicts(a, b):
    return {**a, **b}


def dict_product(d: dict, fixed: dict = None) -> Iterable[Dict]:
    """Generates cartesian product from a dict of parameter ranges.
    So e.g. dic={'A':{1,2},'B':{true,false}} gives
    [{'A': 1, 'B': False},
     {'A': 1, 'B': True },
     {'A': 2, 'B': False},
     {'A': 2, 'B': True }]

    The input dict should contain an iterable as value, but be careful
    to what is the result of iter(value) as for example having a dictionary as
    value will return its keys and not the values!

    So the suggestion is use sets or lists as values in `d`.
    All values from `d` will be coerced to a set to remove duplicates,
    so the order is not guaranteed.

    If the any value in d is not iterable it will be wrapped in a
    list, and thus result equal in all exapanded dicts.  It's also
    possible to pass the `fixed` arg (another dict) whose key/vals
    will be present in all expanded dicts. Be careful that values from `d`
    take precedence in case of duplicate keys.

    """

    fixed = {} if fixed is None else fixed

    assert isinstance(d, dict), "Input `d` must be a dictionary"
    assert isinstance(fixed, dict), "Input `fixed` must be a dictionary"
    ks = d.keys()
    vs = d.values()
    vs = [set(ensure_iterable(x)) for x in vs]
    expanded = product(*vs)
    return (merge_dicts(fixed, dict(zip(ks, tup))) for tup in expanded)
