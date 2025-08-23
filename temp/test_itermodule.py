from jx import nn
from jx.distributed._fsdp import fully_shard
import equinox as eq
import jax
import jax.numpy as jnp
from equinox import field


class M2(eq.Module):
    lin: nn.Linear
    def __init__(self, key):
        self.lin = nn.Linear(4, 4, key=key)

class M1(eq.Module):
    a: M2
    b: list[M2]
    def __init__(self, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.a = M2(k1)
        self.b = [M2(k2), M2(k3)]

print("Original types (all modules):")
for path, t in nn.iter_module_type(M1, include_root=True):
    print(path, t.__name__)




ShardedM1 = nn.transform_module_type_leaves(
    M1,
    lambda path, T: fully_shard(T, "data", min_weight_size=10) if path and path[-1] == "lin" else T,
)

print("\nTransformed type:", ShardedM1.__name__)
print("Transformed types (all modules):")
for path, t in nn.iter_module_type(ShardedM1, include_root=True):
    print(path, t.__name__)

print("\nInstantiate and iterate instances:")
key = jax.random.key(0)

@jax.jit
def init_module():
    inst = ShardedM1(key)
    return inst


inst = init_module()
print(f"DEBUGPRINT[119]: test_itermodule.py:48: inst={inst}")

for path, m in nn.iter_module(inst, include_root=True):
    if hasattr(m, "__predict_out_spec__"):
        print(m, "has predict outspec")
