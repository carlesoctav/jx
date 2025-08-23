from jx._darray import Darray
from . import _functional as functional
from ._dropout import Dropout
from ._embedding import Embedding
from ._layernorm import LayerNorm
from ._linear import Linear
from ._transformation import map_variables, HasPredictMethodOutSpecMixin

import dataclasses
import jax.tree_util as jtu
import jax
from typing import Any, TypeVar




M = TypeVar('M')

is_darray = lambda x: isinstance(x, Darray)

def param_shapes(m: M) -> M:
    def _f(leaf):
        if isinstance(leaf, Darray):
            return Darray(jax.ShapeDtypeStruct(shape = leaf.value.shape, dtype = leaf.value.dtype), names = leaf.pspec)
        else:
            return leaf
    return jtu.tree_map(_f, m, is_leaf = is_darray)


__all__ = [
    "Linear",
    "Embedding",
    "LayerNorm",
    "Dropout",
    "functional",
    "map_variables",
    "param_shapes",
    "HasPredictMethodOutSpecMixin"
]

