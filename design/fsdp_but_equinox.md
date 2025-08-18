**Answers**

- **Per-Submodule Sharding Needed?:** Not mandatory; you only need to shard groups whose full (replicated) size is a material fraction of device memory. Over‑sharding (wrapping every tiny layer) increases all_gather / reduce_scatter latency, compilation cost, and bookkeeping with little peak‑memory win.
- **Is Per-Submodule Feasible in JAX?:** Yes for (a) mostly static graphs (Sequential / feed‑forward blocks) or (b) modules where parameter usage boundaries are clear. Harder (still possible) for highly dynamic control flow or weight sharing because you must ensure each group’s “gather–compute–discard” region is single‑use before next group.
- **Why Wrapper vs `__class__` Mutation (PyTorch-style)?:** 
  - Equinox modules are PyTrees (dataclasses); JAX tracing & caching key off structure+type. Mutating `__class__` mid‑lifecycle risks cache invalidation or silent recompiles.
  - Functional transforms (`jit`, `pjit`, `grad`, `checkpoint`) compose more predictably when wrapping call sites than when altering underlying dataclass identity.
  - Easier explicit opt‑in granularity, clearer debugging, no hidden side effects.
  - Avoids fragile introspection (dataclasses.fields, eqx.filter) and preserves serialization invariants.
  - Encourages pure functional boundary (wrapper manages collectives; inner module stays oblivious).

**Granularity Strategy**

- **Start (Phase 1):** Shard all parameters globally; gather once at start of forward (baseline memory model).
- **Phase 2 (Recommended):** Group large layers (e.g., attention blocks, MLP blocks) so each group fits comfortably when fully gathered (peak memory ≈ max_group_size + sum(sharded_rest)).
- **Phase 3:** User-specified groups (size heuristics: aim each group’s full size ≤ 25–35% of device memory).
- **Skip Micro-Groups:** Layers with < ~0.5–1 MB parameters rarely worth individual gather latency.
- **Heuristic:** Sort param groups by size; form cumulative buckets until threshold (greedy knapsack); leave very small tail merged.

**Core Design Concepts (JAX Approximation of FSDP2)**

- **Persistent Storage (Sharded):** Each parameter stored sharded along data-parallel axis (PartitionSpec('dp', *rest)) where feasible (usually leading dim).
- **Ephemeral Replication:** Before executing a group’s forward, all shards are all_gather’ed to form full weights (lifetime limited to that compute).
- **Backward Gradient Path:** Custom VJP scatters (or slices) summed gradients back to shards (reduce-scatter analogue).
- **Overlap:** Hard to finely pipeline like PyTorch streams; rely on XLA fusion + collective scheduling. Later optimization: prefetch next group’s gather via staged call inside previous group’s compute (experimental).
- **Reshard Policy:** Immediate discard of full weights after group forward to minimize residency (controlled in custom VJP residuals).

**Proposed Minimal API**

```
wrapped = fsdp_wrap(
    module,
    mesh=mesh,
    dp_axis='dp',
    groups=Auto | List[ParamGroupSpec] | Callable[module]->List[ParamGroupSpec],
    shard_policy=DefaultShardPolicy(),
    mixed_precision=FP8 | BF16 | None,
)
```

- `ParamGroupSpec`: names or path filters (regex / predicate) → list of param leaves.
- `shard_policy`: decides which dims of each param can be partitioned (fallback to replicate if not divisible).
- `mixed_precision`: cast full gathered weights to compute dtype; keep sharded master in higher precision.

**Key Internal Components**

- `ParamInfo`: (name, shape, dtype, shardable_dim, original_spec)
- `FSDPGroup`: custom_vjp wrapper implementing gather + compute + gradient scatter.
- `FSDPState`: holds groups metadata, mesh, shard specs, autograd (VJP) handles.
- `ShardInit`: utility to pre-shard params (e.g., pad divisible dimension, then `jax.device_put` with NamedSharding).

**Skeleton Code (Illustrative, Not Production-Ready)**

```python
import dataclasses
from typing import Sequence, Callable, Any
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

@dataclasses.dataclass
class ParamGroupSpec:
    names: tuple[str, ...]  # parameter path names (e.g. "layers_3_weight")
    # future: prefetch flag, precision override, etc.

def _axis_index(axis):
    return lax.axis_index(axis)

def shard_param(param, mesh: Mesh, dp_axis: str):
    # Simple heuristic: shard leading dimension if divisible
    if param.ndim >= 1 and param.shape[0] % mesh.shape[dp_axis] == 0:
        spec = P(dp_axis, *([None] * (param.ndim - 1)))
    else:
        spec = P()  # replicate
    return jax.make_array_from_callback(
        param.shape,
        NamedSharding(mesh, spec),
        lambda idx: param[idx]  # naive; real impl: re-layout once
    )

def _gather_full(shard, axis):
    # shard shape: [local_chunk, ...]; gather -> [axis_size, local_chunk, ...] then reshape
    axis_size = lax.psum(1, axis)
    gathered = lax.all_gather(shard, axis, axis=0)  # shape [axis_size, local_chunk, ...]
    new_leading = gathered.shape[0] * gathered.shape[1]
    full_shape = (new_leading,) + gathered.shape[2:]
    return gathered.reshape(full_shape)

def _scatter_grad(full_grad, original_shard_shape, axis):
    axis_idx = _axis_index(axis)
    shard_len = original_shard_shape[0]
    start = axis_idx * shard_len
    sliced = lax.dynamic_slice_in_dim(full_grad, start, shard_len, axis=0)
    # Sum across data-parallel axis (reduce part)
    summed = lax.psum(sliced, axis)
    return summed  # shape == original shard

def make_group_vjp(apply_fn, params_tuple, axis_name):
    # params_tuple: tuple of sharded params for this group
    @jax.custom_vjp
    def run(inputs, *group_params):
        full_params = tuple(_gather_full(p, axis_name) for p in group_params)
        return apply_fn(full_params, inputs)

    def fwd(inputs, *group_params):
        full_params = tuple(_gather_full(p, axis_name) for p in group_params)
        out = apply_fn(full_params, inputs)
        # Save shapes only (not full params) to minimize residual
        meta = tuple(p.shape for p in group_params)
        return out, (meta,)
    def bwd(residual, g_out):
        (shapes,) = residual
        # We do NOT have full params; gradient rules inside apply_fn
        # produce gradients wrt full_params implicitly by re-running?
        # Simplification: assume apply_fn is pure & recompute (cheap recompute variant)
        # Real impl would capture primals needed.
        raise NotImplementedError("Fill with recompute logic or capture jaxpr.")
    run.defvjp(fwd, bwd)
    return run

@dataclasses.dataclass
class FSDPModule(eqx.Module):
    inner: eqx.Module
    group_specs: tuple[ParamGroupSpec, ...]
    mesh: Any
    dp_axis: str = 'dp'

    def __call__(self, *args, **kwargs):
        # Prototype naive path: run full replicate (Phase 1)
        def apply_all(params, *a, **kw):
            return self.inner(*a, **kw)

        return eqx.filter_jit(apply_all)(eqx.filter(self.inner, eqx.is_array), *args, **kwargs)

def fsdp_wrap(module: eqx.Module,
              mesh: Mesh,
              groups: Sequence[ParamGroupSpec] | None = None,
              dp_axis: str = 'dp') -> FSDPModule:
    # TODO: extract param names, build default grouping if groups is None
    if groups is None:
        groups = [ParamGroupSpec(tuple())]  # sentinel
    return FSDPModule(module, tuple(groups), mesh, dp_axis)
```

Notes:
- Backward path above shows where a real implementation would either (a) capture a closed-over jaxpr for `apply_fn` and use `jax.vjp` with full params reconstructed on-the-fly, or (b) retain lightweight compression of full params (e.g., quantized) in residual if recompute cost is prohibitive.
- Production version should rely on `psum_scatter` once available (gives summed shards directly) instead of manual slice+psum.
- Sharding initialization should re-layout arrays once (not per-call) using `jax.device_put` with `NamedSharding`.

**Example Usage (Conceptual)**

```
devices = jax.devices()
mesh = Mesh(mesh_utils.create_device_mesh((len(devices),)), ('dp',))

model = MLP(...)  # Equinox module
wrapped = fsdp_wrap(model, mesh)

@jax.pjit(in_shardings=(P('dp'),), out_shardings=P('dp'))
def train_step(state, batch):
    def loss_fn(params, batch):
        preds = wrapped.apply(params, batch.x)  # once integrated with param passing
        return cross_entropy(preds, batch.y)
    grads = jax.grad(loss_fn)(state.params, batch)
    state = state.apply_gradients(grads=grads)
    return state
```

Memory effect: With sharded storage, each device holds roughly (1 / dp_world_size) of large weight matrices (for shardable leading dims) plus transient full copies only for active group during forward.

**Handling Non-Sequential Graphs**

- Build param→usage mapping via ahead-of-time trace (e.g., run a dummy forward with a wrapper that records which params were accessed between barrier markers).
- Post-process into groups minimizing overlaps (graph coloring / interval partitioning).
- Insert synthetic boundaries via wrapper calls to trigger gather/release.

**Limitations / Open Points**

- Gradient scatter path not fully coded (placeholder `bwd`).
- Real gather must consider parameter padding and non-leading-dim sharding.
- Compiler scheduling of collectives may duplicate gathers if not isolated; prefer encapsulating group forward in its own `pjit` or using `named_call`.
- Mixed precision: cast only full (ephemeral) replicas; keep shards in higher precision master for stability.

**Next Steps**

- Implement real param path extraction (using `jax.tree_util` + Equinox naming).
- Finish custom VJP with recompute or capture logic.
- Add heuristic auto-grouping (size threshold).
- Benchmark gather/scatter latency vs memory saved.
- Add mixed precision and optional prefetch (launch next group all_gather early).

Let me know if you’d like me to flesh out the custom VJP backward, auto-grouping heuristic code, or a working end-to-end notebook example next.
