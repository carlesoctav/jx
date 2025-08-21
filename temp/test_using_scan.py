import equinox as eq
import jax
from equinox import nn, field
from flax import linen

#TLDR bad idea
class MLPLayer(eq.Module):
    layers: list
    n: field(static = True) 

    def __init__(
        self,
        n: int = 10,
        key = None,
    ):
        keys = jax.random.split(key, n)
        self.n = n
        self.layers = [nn.Linear(i, i+1, key = keys[i-1]) for i in range(1, n)] 
 
    def __call__(
        self,
        x: jax.Array
    ):
        xs = [eq.partition(self.layers[i], eq.is_array) for i in range(len(self.layers))] 
        def step(carry, model_def):
            model = eq.combine(*model_def)
            output = model(carry)
            return output, None  
        output, _ = jax.lax.scan(step, init = x , xs = xs) 
        return output
        # for layer in self.layers:
        #      x = layer(x)
        #
        # return x



key = jax.random.key(32)

mlp = MLPLayer(10, key)
input_key = jax.random.split(key)[0]
x = jax.random.normal(input_key, (1,))
output = mlp(x)
print(f"DEBUGPRINT[101]: test_using_scan.py:34: output={output}")
