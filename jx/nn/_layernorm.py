import equinox as eq
import jax
import jax.numpy as jnp
from equinox import field
from jax import lax
from jax.nn.initializers import ones as ones_init, zeros as zeros_init
from jax.sharding import PartitionSpec as P
from jaxtyping import DTypeLike, PRNGKeyArray

from jx import Darray


Array = jax.Array


class LayerNorm(eq.Module):
    weight: Darray | None
    bias: Darray | None
    normalized_shape: tuple[int, ...] = field(static=True)
    eps: float = field(static=True)
    elementwise_affine: bool = field(static=True)
    dtype: DTypeLike = field(static=True)

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        *,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        weight_pspec: P | None = None,
        bias_pspec: P | None = None,
        dtype: DTypeLike = jnp.float32,
        key: PRNGKeyArray | None = None,
    ):
        self.elementwise_affine = elementwise_affine
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else normalized_shape
        self.eps = eps
        self.dtype = dtype

        if key is None:
            key = jax.random.PRNGKey(0)


        if self.elementwise_affine:
            wkey, bkey = jax.random.split(key, 2)
            wvalue = ones_init(wkey, normalized_shape, dtype=dtype)
            self.weight = Darray(wvalue, weight_pspec)
            if bias:
                bvalue = zeros_init(bkey, normalized_shape, dtype=dtype)
                self.bias = Darray(bvalue, bias_pspec)
            else:
                self.bias = None
        else:
            self.weight = None
            self.bias = None


    def __call__(
        self,
        x: Array,
        *,
        rngs: PRNGKeyArray | None = None
    ) -> Array:
        if x.shape != self.normalized_shape:
            raise ValueError(
                f"Input shape {x.shape} does not match normalized shape {self.normalized_shape}"
                
            )
        mean = jnp.mean(x, keepdims=True) 
        var = jnp.var(x, keepdims=True)
        var = jnp.maximum(var, 0.0)
        inv = lax.rsqrt(var + self.eps)  # Fixed: should be + not -
        y = (x - mean) * inv  # Fixed: should be * not /

        if self.weight is not None:
            y = self.weight.value * y
        if self.bias is not None:
            y = y + self.bias.value

        return y


