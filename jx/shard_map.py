import functools as ft
import jax
from jax import shard_map
import jax.tree_util as jtu
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import PyTree
from ._filters import is_array, partition, combine
from ._module import Module, module_update_wrapper, Partial, Static
from ._custom_types import sentinel  # like elsewhere

def _flatten_arrays(pytree):
    leaves, treedef = jtu.tree_flatten(pytree)
    array_mask = [is_array(x) for x in leaves]
    array_leaves = [x for x, m in zip(leaves, array_mask) if m]
    # Build static meta structure with arrays replaced by sentinel None
    static_leaves = [None if m else x for x, m in zip(leaves, array_mask)]
    static_meta = (tuple(static_leaves), treedef)
    return array_leaves, static_meta, array_mask, treedef

def _reconstruct(static_meta, array_leaves):
    static_leaves, treedef = static_meta
    it = iter(array_leaves)
    rebuilt = [next(it) if s is None else s for s in static_leaves]
    return jtu.tree_unflatten(treedef, rebuilt)

class _ShardMapWrapper(Module):
    _fun: callable
    _mesh: Mesh
    _in_specs: PyTree
    _out_specs: PyTree
    _check_rep: bool
    _auto: frozenset
    _cached: dict  # (structure keys) -> shard_map fn

    @property
    def __wrapped__(self): return self._fun

    def __call__(self, /, *args):
        # Disallow kwargs for parity (or extend)
        array_leaves, static_args_meta, array_mask, args_treedef = _flatten_arrays(args)
        # Derive dynamic in_specs tuple aligned to array_leaves
        full_in_specs = self._in_specs
        # Broadcast prefix: follow filter() prefix approach
        # simplest: map a lambda over args structure:
        def _expand(spec_sub, arg_sub):
            if isinstance(spec_sub, (PartitionSpec, type(None))):
                return jtu.tree_map(lambda _: spec_sub, arg_sub)
            return spec_sub  # already subtree
        expanded_in = jtu.tree_map(_expand, full_in_specs, args, is_leaf=lambda x: isinstance(x, (PartitionSpec, type(None))))
        flat_specs, _ = jtu.tree_flatten(expanded_in)
        dynamic_in_specs = [s for s, m in zip(flat_specs, array_mask) if m]
        # Cache key
        key = (args_treedef, tuple(array_mask))
        try:
            sm = self._cached[key]
        except KeyError:
            def inner(array_tuple):
                full_args = _reconstruct(static_args_meta, array_tuple)
                out = self._fun(*full_args)
                out_array_leaves, out_static_meta, out_array_mask, _ = _flatten_arrays(out)
                return tuple(out_array_leaves), Static(out_static_meta)
            # Need one example output spec tuple: flatten user out_specs similarly
            # (omitted for brevity; analogous to in_specs)
            sm = shard_map(
                inner,
                mesh=self._mesh,
                in_specs=tuple(dynamic_in_specs),
                out_specs=( ...dynamic_out_specs_tuple..., None),
                check_rep=self._check_rep,
                auto=self._auto,
            )
            self._cached[key] = sm
        out_arrays, static_meta_wrapper = sm(tuple(array_leaves))
        out = _reconstruct(static_meta_wrapper.value, out_arrays)
        return out

def filter_shard_map(
    fun=sentinel, *, mesh, in_specs, out_specs, check_rep=True, auto=frozenset()
):
    if fun is sentinel:
        return ft.partial(
            filter_shard_map,
            mesh=mesh, in_specs=in_specs, out_specs=out_specs,
            check_rep=check_rep, auto=auto
        )
    wrapper = _ShardMapWrapper(
        _fun=fun,
        _mesh=mesh,
        _in_specs=in_specs,
        _out_specs=out_specs,
        _check_rep=check_rep,
        _auto=auto,
        _cached={}
    )
    return module_update_wrapper(wrapper)
