import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jx._darray import Darray
from jx.nn import Linear, map_variables


def convert_to_bit(
    tree
):
    def f(x):
        if x.value.ndim >= 2:
            return Darray(jnp.triu(x.value), None)
        else:
            return x
    return jtu.tree_map(f, tree, is_leaf = lambda x: isinstance(x, Darray))


BitLinear = map_variables(
    Linear,
    map_in_fn=convert_to_bit,
    map_name = "BitLinear",
    mutate=True
)

bit_linear = BitLinear(
    in_features = 10,
    out_features = 10,
    key = jax.random.key(90)
)

w = bit_linear.weight.value
print(f"DEBUGPRINT[91]: test_map_variables.py:24: w={w}")
key = jax.random.key(100)

x = jax.random.normal(key, (1, 10))

output, bit_linear= jax.vmap(bit_linear)(x)
print(f"DEBUGPRINT[92]: test_map_variables.py:34: output={output}")

print(bit_linear.weight.value)
