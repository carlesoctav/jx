
import equinox as eq
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import PRNGKeyArray

from jx._darray import first_from


Array = jax.Array


class Dropout(eq.Module):
    """Dropout layer for regularization.
    
    Randomly sets elements to zero during training and scales remaining elements.
    Works on single samples without batch dimension - use jax.vmap for batched operations.
    
    During inference (when key=None), acts as identity function.
    During training (when key is provided), applies dropout with given probability.
    
    Args:
        p: Probability of an element being zeroed (default: 0.5)
        inference: If True, always acts as identity (no dropout)
    """
    p: float 
    inference: bool 

    def __init__(
        self,
        p: float = 0.5,
        *,
        inference: bool = False,
    ):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Dropout probability must be between 0 and 1, got {p}")
        
        self.p = p
        self.inference = inference

    def __call__(
        self, 
        x: Array, 
        *, 
        key: PRNGKeyArray | None = None,
        inference: bool | None = None
    ) -> Array:

        inference = first_from(
            inference,
            self.inference,
            error_msg="""No `inference` argument was provided to Dropout 
                as either a __call__ argument or class attribute""",
        )

        
        if inference: 
            return x 

        if not inference and key is None:
            raise RuntimeError(
                "Dropout requires a key when running in non-inference mode."
            )

        if self.p == 1.0:
            return jnp.zeros_like(x)
        
        keep_prob = 1.0 - lax.stop_gradient(self.p) 
        mask = jax.random.bernoulli(key, keep_prob, shape=x.shape)
        output = jnp.where(mask, x / keep_prob, 0.0)
        
        return output


