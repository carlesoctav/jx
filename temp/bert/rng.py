import jax
from jax.nn.initializers import normal


key = jax.random.PRNGKey(42)
print(f"DEBUGPRINT[25]: rng.py:4: key={key}")

new_key = jax.random.split(key, 1)
print(f"DEBUGPRINT[32]: rng.py:8: new_key={new_key}")
default = normal()
rand = default(new_key, (10,10))
print(f"DEBUGPRINT[29]: rng.py:10: rand={rand}")

