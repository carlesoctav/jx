import jax
from jx.layer import  Linear


key = jax.random.key(100)
a = Linear.init(10, 5, key = key)
x = jax.random.normal(key, (3, 10))
y = jax.vmap(a)(x)
print(f"DEBUGPRINT[6]: test_linear.py:8: y={y}")

