import functools as ft
import numpy as np
from typing import Any, Callable

import jax
import jax.tree_util as jtu
from equinox import eq
import logging
from jx._darray import Darray
from jx import nn


Pytree = Any


is_darray = lambda x: isinstance(x, Darray)


@jax.named_scope("gather_params_2")
def gather_array_with_mean_grads(value: jax.Array, axis: int, axis_name: str):
    axis_size = jax.lax.psum(1, axis_name)

    @jax.custom_gradient
    def _f(x):
        def _grad_fn(g):
            return jax.lax.psum_scatter(g, axis_name, scatter_dimension = axis, tiled = True) / axis_size

        gathered = jax.lax.all_gather(value, axis_name = axis_name, axis = axis, tiled = True)

        return gathered, _grad_fn
    
    return _f(value)

@jax.named_scope("gather_params")
def gather_params(params: Pytree, axis_name: str):
    def _gather(p: Darray | Any):
        if isinstance(p, Darray) and p.names and axis_name in p.names:
            value, names = p.value, p.names
            shard_axis = names.index(axis_name) 
            gathered = gather_array_with_mean_grads(value, axis = shard_axis, axis_name = axis_name)
            new_names = names[:shard_axis] + (None,) + names[shard_axis+1:]
            return Darray(gathered, new_names)
        else:
            return p
    return jtu.tree_map(_gather, params, is_leaf = is_darray)


@jax.named_scope("shard_params") 
def shard_params(model, axis_name: str, min_weight_size = 2**18):
    """
    create a partition_spec for the shard params
    this is useful for inspecs and outspecs for shard_amp, or pmap function
    """
    params = eq.filter(model, eq.is_array)
    axis_idx = jax.lax.axis_index(axis_name)
    axis_size = jax.lax.psum(1, axis_name)
    def _shard(p: Darray | Any):
        if isinstance(p, Darray):
            value, names = p.value, p.names

            if p.names is None:
                names = (None,) * value.ndim

            if len(names) != value.ndim:
                raise ValueError("attribute names on Darray should have the same dimension with the attribute value")

            if axis_name in names:
                logging.warning(
                    f"Parameter {value.shape} with names {names} already sharded on axis {axis_name}."
                )
                return p
            elif value.size <= min_weight_size:
                logging.info(
                    f"Parameter {value.shape} with names {names} too small to shard, size {value.size} < {min_weight_size}."
                )
                return p
            else:
                shape = value.shape
                idx = np.argsort(shape)[::-1]
                for i in idx:
                    if shape[i] % axis_size == 0 and names[i] is None:
                        split_size = shape[i] // axis_size
                        p_sharded = Darray(
                            value = jax.lax.dynamic_slice_in_dim(
                                value, axis_idx * split_size, split_size, axis = i
                            ),
                            names = names[:i] + (axis_name,) + names[i+1:],
                        )
                        return p_sharded
                logging.warning(
                    f"Could no shard {value.shape} with names {names} on axis {axis_name}, no suitable axis found"
                )
        else:
            return p


    return jtu.tree_map(_shard, params, is_leaf = is_darray)


def fully_shard(
    target: type[nn.Module],
    model_axis_name: str,
    min_weight_size = 2**18,
    *,
    where = eq.is_array,
)-> type[nn.Module]:
    """Wrap a Module so its parameters are gathered just-in-time.

    Parameters stay sharded at rest; before every call we materialise the
    full parameters (via gather_params) inside a temporary copy of the
    module. A custom gradient on the gather op scatters grads back to the
    local shards, so we do not need an explicit post-call reshard step.

    This keeps the forward call signature unchanged (returns only the
    original module output) and avoids threading `(out, new_model)` through
    the model. Memory use matches the previous mutate=True version because
    the full parameters had to exist during the forward anyway; we simply
    skip reconstructing a new sharded Module afterwards.
    """
    return nn.map_variables(
        target,
        map_in_fn = ft.partial(gather_params, axis_name = model_axis_name),
        map_name = "FSDP",
        where= where,
        mutate = False,  # don't change call signature
    )

