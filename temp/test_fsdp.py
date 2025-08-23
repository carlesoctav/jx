import logging
import os
import subprocess
import sys
from collections.abc import Callable
from functools import partial
from typing import Any

import equinox as eqx
import jax
from jax.experimental.mesh_utils import create_device_mesh
import jax.tree_util as jtu
import numpy as np
from equinox import field, module_update_wrapper
from jax import lax
from jx import Darray
from jx.nn import Linear
from jax.sharding import Mesh


PyTree = Any
Parameter = Darray | jax.Array 


def set_XLA_flags_gpu():
    flags = os.environ.get("XLA_FLAGS", "")
    flags += (
        "--xla_gpu_enable_triton_softmax_fusion=true "
        "--xla_gpu_triton_gemm_any=false "
        "--xla_gpu_enable_async_collectives=true "
        "--xla_gpu_enable_latency_hiding_scheduler=true "
        "--xla_gpu_enable_highest_priority_async_stream=true "
    )
    os.environ["XLA_FLAGS"] = flags


def simulate_CPU_devices(device_count: int = 8):
    flags = os.environ.get("XLA_FLAGS", "")
    flags += f" --xla_force_host_platform_device_count={device_count}"
    os.environ["XLA_FLAGS"] = flags
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        import ml_collections
    except ImportError:
        install_package("ml_collections")


def install_package(package: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])


simulate_CPU_devices()

class _MapVariableWrapper(eqx.Module):
    module: eqx.Module
    map_in: Callable = field(static=True)
    map_out: Callable | None = field(static=True, default=None)

    @property
    def __wrapped__(self):
        return self.module

    def __call__(self, *args, **kwargs):
        # Apply map_in to every Darray leaf (gather full params) before forward.
        def apply(fn, m):
            if fn is None:
                return m
            return jtu.tree_map(lambda x: fn(x) if isinstance(x, Darray) else x,
                                 m,
                                 is_leaf=lambda x: isinstance(x, Darray))
        module_in = apply(self.map_in, self.module)
        out = module_in(*args, **kwargs)
        # Re-shard after forward for storage.
        module_out = apply(self.map_out, module_in)
        object.__setattr__(self, "module", module_out)
        return out

    # def __call__(self, *args, **kwargs):
    #     params, non_params = eqx.partition(self.module, eqx.is_array)
    #     params_in = jtu.tree_map(self.map_in, params, is_leaf=lambda x: isinstance(x, Darray))
    #     module_in = eqx.combine(params_in, non_params)
    #     out = module_in(*args, **kwargs)
    #     if self.map_out is not None:
    #         params_out = jtu.tree_map(self.map_out, params_in, is_leaf=lambda x: isinstance(x, Darray))
    #         module_out = eqx.combine(params_out, non_params)
    #         object.__setattr__(self, "module", module_out)
    #     else:
    #         object.__setattr__(self, "module", module_in)
    #     return out

def map_variables(
    module: eqx.Module,
    trans_in_fn,
    trans_out_fn = None,
):
    return module_update_wrapper(_MapVariableWrapper(module, trans_in_fn, trans_out_fn))

@jax.named_scope("shard_params")
def shard_params(params: PyTree, axis_name: str, min_weight_size: int = 2**18) -> PyTree:
    """Shard parameters across the given mesh axis.

    Args:
        params: The parameters to shard.
        axis_name: The axis to shard parameters across.
        min_weight_size: The minimum size of a parameter to shard. Parameters with fewer values will not be sharded.

    Returns:
        PyTree of same structure as params, but with leaves sharded over new axis if possible.
    """
    axis_idx = jax.lax.axis_index(axis_name)
    axis_size = jax.lax.psum(1, axis_name)

    def _split(x: Parameter) -> Parameter:
        # Skip non-array / placeholder leaves (e.g. None) produced by eqx.partition
        if x is None or not isinstance(x, (Darray, jax.Array)):
            return x
        if isinstance(x, Darray):
            value, names = x.value, x.pspec
        else:
            value = x
            names = (None,) * value.ndim
        if names is None:
            return x
        if axis_name in names:
            logging.warning(
                f"Parameter {value.shape} with names {names} already sharded on axis {axis_name}."
            )
            return x
        elif value.size <= min_weight_size:
            logging.info(
                f"Parameter {value.shape} with names {names} too small to shard, size {value.size} < {min_weight_size}."
            )
            return x
        else:
            shape = value.shape
            idx = np.argsort(shape)[::-1]  # Shard along largest possible axis.
            for i in idx:
                if shape[i] % axis_size == 0 and names[i] is None:
                    split_size = shape[i] // axis_size
                    p_sharded = Darray(
                        value=lax.dynamic_slice_in_dim(  # Shard to keep on present device.
                            value, axis_idx * split_size, split_size, axis=i
                        ),
                        pspec=names[:i] + (axis_name,) + names[i + 1 :],
                    )
                    return p_sharded
            logging.warning(
                f"Could not shard {value.shape} with names {names} on axis {axis_name}, no suitable axis found."
            )
            return x

    return jax.tree_util.tree_map(
        _split,
        params,
        is_leaf=lambda x: isinstance(
            x, Darray
        ),  # Consider a nn.Partitioned object as a leaf.
    )


def gather_array_with_mean_grads(x: jax.Array, axis: int, axis_name: str):
    """Gathering with averaging gradients across replicas."""
    axis_size = jax.lax.psum(1, axis_name)

    @jax.custom_gradient
    def f(x):
        def grad_fn(g):
            return (
                jax.lax.psum_scatter(g, axis_name, scatter_dimension=axis, tiled=True) / axis_size
            )

        return jax.lax.all_gather(x, axis_name, axis=axis, tiled=True), grad_fn

    return f(x)

@jax.named_scope("gather_params")
def gather_params(params: PyTree, axis_name: str) -> PyTree:
    """Gather parameters from all replicas across the given axis.

    Leaves that are sharded on `axis_name` are all-gathered (with mean grads) and their
    corresponding axis name is replaced by None. Unsharded leaves are returned unchanged.
    """
    def _gather(p: Parameter) -> Parameter:
        if isinstance(p, Darray) and p.pspec is not None and axis_name in p.pspec:
            names = p.pspec
            shard_axis = names.index(axis_name)
            gathered = gather_array_with_mean_grads(p.value, axis=shard_axis, axis_name=axis_name)
            new_names = names[:shard_axis] + (None,) + names[shard_axis + 1:]
            return Darray(gathered, new_names)
        else:
            return p  # Leave unchanged (still a Darray)
    return jax.tree_util.tree_map(_gather, params, is_leaf=lambda x: isinstance(x, Darray))


is_darray = lambda x: isinstance(x, Darray)

def get_partition_spec(
    tree
):
    def f(x):
        if is_darray(x) and x.names:
            return jax.P(*x.names)
        else:
            return jax.P(None) 

    return jtu.tree_map(f, tree, is_leaf = is_darray)

key = jax.random.key(10)
len(jax.devices())
devices = create_device_mesh((8,) ,jax.devices())
mesh = Mesh(devices, axis_names = ("data",))

linear = Linear(10, 960, key = key, weight_pspec=("data", None))


gather = partial(gather_params, axis_name="data")
shard = partial(shard_params, axis_name="data", min_weight_size=100)


linear = map_variables(linear, gather, shard)
pspec = get_partition_spec(linear)
print(f"DEBUGPRINT[89]: test_fsdp.py:221: pspec={pspec}")


@partial(jax.shard_map, in_specs=(pspec, jax.P("data", None)), out_specs=jax.P("data", None), mesh=mesh)
def train(model, batch):
    def get_val(w):
        return w.value if isinstance(w, Darray) else w

    w_local = get_val(model.module.weight)
    print(f"DEBUGPRINT[90]: test_fsdp.py:232: w_local={w_local.shape}")
    out = jax.vmap(model)(batch)
    return out



batch = jax.random.normal(key, (80, 10))

out1 = train(linear, batch)
out2 = jax.vmap(linear.module)(batch)

# Safe access after execution (outside shard_map tracing)
weight_after = linear.module.weight.value
named_sharding = jax.NamedSharding(mesh, spec =jax.P("data", None))
weight_after = jax.device_put(weight_after, named_sharding)
jax.debug.visualize_array_sharding(weight_after)




