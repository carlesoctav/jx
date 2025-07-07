import jax
from jx.core._src import Darray


a = jax.numpy.eye(19, 19)
t = Darray(a)
print(f"DEBUGPRINT[3]: test_tensor.py:6: t={t}")
