import jax
import jax.numpy as jnp
from jax import Array
import equinox as eq
from jx import Darray
from jx import nn


class LinearButRandomlyBecomeZero(eq.Module):
    linear: nn.Linear

    def __call__(self, x: Array):
        """Applies the wrapped linear layer and returns (output, new_module).

        We create a new module with its weight zeroed (pure/functional style) and
        return it so the caller can rebind. This is the JAX/Eqinox idiom instead
        of mutating in place.
        """
        out = self.linear(x)
        new_self = eq.tree_at(
            lambda m: m.linear.weight,
            self,
            Darray(jnp.zeros_like(self.linear.weight.value))
        )
        return out, new_self


key = jax.random.key(10)
linear = nn.Linear(10, 1, key=key)
linear_random = LinearButRandomlyBecomeZero(linear=linear)
print("Before call: weight sum =", linear_random.linear.weight.value.sum())

x = jax.random.normal(key, (10,))
output, linear_random = linear_random(x)  # Rebind to updated module
print("After call:  weight sum =", linear_random.linear.weight.value.sum())
print("Output:", output)
