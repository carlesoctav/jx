from typing import Literal
import equinox as eq

import typing as tp
import jax.numpy as jnp
import jax
from jax.sharding import PartitionSpec as P
from jaxtyping import DTypeLike, PRNGKeyArray
from jx import Darray
from equinox import field
from jax.nn.initializers import lecun_uniform, zeros as zeros_init

Array = jax.Array

default_init = lecun_uniform()


class Linear(eq.Module):
    weight: Darray
    bias: Darray | None
    in_features: int = field(static=True)
    out_features: int = field(static=True)
    bias_enabled: bool = field(static=True)
    dtype: DTypeLike = field(static = True)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        weight_pspec = None,
        bias: bool = True,
        dtype: DTypeLike | None = None,
        key: PRNGKeyArray,
    ):

        wkey, bkey = jax.random.split(key, 2)
        wvalue = default_init(wkey, (out_features, in_features), dtype=dtype)
        self.weight = Darray(wvalue, weight_pspec)

        if bias:
            bvalue = zeros_init(bkey, (out_features,), dtype=dtype)
            self.bias = Darray(bvalue)
        else:
            self.bias = None

        self.in_features = in_features
        self.out_features = out_features
        self.bias_enabled = bias
        self.dtype = dtype


    def __call__(
        self,
        inputs: Array,
        *, 
        rngs: PRNGKeyArray | None = None
    ) -> Array:
        w = self.weight.value if hasattr(self.weight, 'value') else self.weight
        y =  w @ inputs #(out feat, in_feat) (in_feat, )
        if self.bias is not None:
            b = self.bias.value if hasattr(self.bias, 'value') else self.bias
            y = y + b
        return y

#need to run under shard_map?
class TPLinear(eq.Module):
    dense_fn: Linear 
    model_axis_name: str
    tp_mode: Literal["scatter", "gather", "none"] = "none"
    pass


    def __init__(
        in_features: int,
        out_features: int,
        *,
        weight_pspec: P | None = None,
        bias: bool = True,
        dtype: DTypeLike | None = None,
        key: PRNGKeyArray,
    ):
        pass
