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


#TODO: dtype is usselss for now
class Linear(eq.Module):
    weight: Darray
    bias: Darray | None
    in_features: int = field(static=True)
    out_features: int = field(static=True)
    use_bias: bool = field(static=True)
    dtype: DTypeLike = field(static = True)
    param_dtype: DTypeLike = field(static = True)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        weight_pspec: tp.Optional[P] = None,
        use_bias: bool = True,
        dtype: tp.Optional[DTypeLike] = None,
        param_dtype: DTypeLike = jnp.float32,
        key: PRNGKeyArray,
    ):

        wkey, bkey = jax.random.split(key, 2)
        wvalue = default_init(wkey, (out_features, in_features), param_dtype)
        self.weight = Darray(wvalue, weight_pspec)

        if use_bias:
            bvalue = zeros_init(bkey, (out_features,), dtype=param_dtype)
            self.bias = Darray(bvalue)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.dtype = dtype or self.weight.value.dtype
        self.param_dtype = param_dtype 


    def __call__(self, inputs: Array) -> Array:
        w = self.weight.value
        y =  w @ inputs #(out feat, in_feat) (in_feat, )
        if self.bias:
            b = self.bias.value
            y = y + b
        return y
