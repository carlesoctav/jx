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
    use_bias: bool = field(static=True)

    @classmethod
    def init(
        cls,
        in_features: int,
        out_features: int,
        *,
        weight_pspec: tp.Optional[P] = None,
        use_bias: bool = True,
        param_dtype: DTypeLike = jnp.float32,
        key: PRNGKeyArray,
    ):

        wkey, bkey = jax.random.split(key, 2)
        wvalue = default_init(wkey, (out_features, in_features), param_dtype)
        weight = Darray(wvalue, weight_pspec)

        if use_bias:
            bvalue = zeros_init(bkey, (out_features,), dtype=param_dtype)
            bias = Darray(bvalue)

        return cls(
            weight=weight,
            bias=bias if use_bias else None,
            in_features=in_features,
            out_features=out_features,
            use_bias=use_bias,
        )

    def __call__(self, inputs: Array) -> Array:
        w = self.weight.value
        y =  w @ inputs #(out feat, in_feat) (in_feat, )
        if self.bias:
            b = self.bias.value
            y = y + b
        return y
