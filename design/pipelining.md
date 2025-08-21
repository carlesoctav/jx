You asked two things:

1) In Flax notebook: Why do we wrap `PipelineModule` inside `ModelParallelismWrapper` and why the unstack → work → stack parameter dance?
2) How to implement a similar pipeline parallelism mechanism in Equinox, but without externally wrapping the main module—i.e. by transforming the `type[eqx.Module]` (a class-level / decorator style transformation) instead of adding a separate wrapper module.

Below I first explain the Flax design choices, then show two progressively more advanced Equinox implementations: (A) simplest conceptual pipeline (replicated params) and (B) a sharded/stage-stacked variant; finally (C) a decorator that returns a pipelined subclass (a “type transformation”).

--------------------------------------------------
**Why Wrap PipelineModule with ModelParallelismWrapper**

- Separation of concerns: `PipelineModule` only implements the pipeline schedule (micro‑batch loop + `ppermute`). `ModelParallelismWrapper` injects model-axis parameter sharding, RNG folding, and optional masking (zeroing unused replicas).
- Uniform parameter annotation: Flax uses variable collections; to assign a `PartitionSpec('model', …)` uniformly, the wrapper applies `nn.map_variables` so every param (including those created inside the scanned MLP layers) gains a leading model axis.
- Initialization differences per stage: Folding the `params` RNG across the model axis inside the wrapper ensures each stage gets distinct initialization, instead of all stages sharing identical weights.
- Masking input/output layers: The wrapper’s `mask_except_model_idx` zeros unused replicas so optimizer comms do not have to all-gather/sync those weights unnecessarily.
- stack_params/unstack_params: `unstack_params` removes the synthetic model axis before calling the “real” module so internal logic sees ordinary shapes; after applying the module, `stack_params` reintroduces a leading model dimension plus `Partitioned` annotations for pjit/shard_map. This lets you (a) reuse unmodified module code and (b) present properly partitioned, axis‑named leaves to JAX’s partitioner.

--------------------------------------------------
**What stack_params / unstack_params Achieve**

- Shape adaptation: Adds/removes a leading dimension whose length = number of model stages. That dimension represents the stage shard of each parameter.
- Sharding metadata: Wraps leaves in `nn.Partitioned` with names including `'model'`, enabling automatic layout inference.
- Masking: When masking, only one slice retains original values; others become zeros so their gradients remain zero—no cross-device parameter sync needed for those “dummy” replicas.
- Symmetry for transforms: Because `nn.scan` (over layers or microbatches) and `map_variables` interact, keeping the single‑device logical shape during computation greatly simplifies internal code; stacking occurs only at boundaries.

--------------------------------------------------
**Equinox vs Flax: Core Differences Relevant Here**

- No variable collections: Equinox modules are plain pytrees; parameters are just leaves— no post-hoc mutation via `map_variables`. You instead transform the tree directly (e.g. with `eqx.tree_at` or `jax.tree_map`).
- Simpler RNG handling: You pass keys explicitly; no collection-level RNG splitting. This makes per-stage or per-microbatch key derivation explicit.
- Sharding annotations: Equinox doesn’t have a built-in `Partitioned` wrapper; you rely on `pjit`/named sharding and shapes. So to mimic “stack/unstack” you either (a) physically add a leading stage axis to parameter arrays and assign a `PartitionSpec('model', ...)`, or (b) split the module into per-stage submodules and rely on partitioning at module granularity.
- No metaclass magic needed: A “type transformation” is naturally expressed as a decorator returning a subclass with an overridden `__call__`.

You requested a transformation at the class/type level rather than a wrapper instance; below I show a decorator-based pattern.

--------------------------------------------------
**Design Options in Equinox**

- Option A (Replica + Conditional Execution): Keep full module replicated; each device runs only its subset of layers (others become no‑ops). Easiest to implement; wastes memory.
- Option B (Stage-Stacked Parameters): Add leading stage dimension to each parameter, mask unused stages with zeros, index by `lax.axis_index('model')`. Lets `PartitionSpec('model', …)` shard the first dimension so each device stores only its slice (once you use `pjit` with a mesh).
- Option C (Physical Split): Restructure into `Stage` submodules inside a `Pipelined` container; with `pjit` you can shard container fields (more manual).
- Option D (Decorator / Class Transform): A decorator returns a subclass that inserts pipeline scheduling into `__call__`, precomputes stage boundaries, and (optionally) stage-stacks parameters at construction.

Below I give working skeletons for A, B, and the decorator for D; you can mix B + D.

--------------------------------------------------
**Core Pipeline Scheduling Logic (Equinox Version)**

We mirror the Flax logic:
- Reshape batch into `num_microbatches`.
- Append dummy microbatches (`num_stages - 1`) to flush pipeline.
- `lax.scan` over (microbatches + dummies).
- Per step:
  - Stage 0 consumes a real microbatch; others consume the ppermute’d state.
  - Run stage-local layers.
  - Emit output only on last stage; zeros elsewhere.
  - `ppermute` the activation to next stage.
- Collect last `num_microbatches` true outputs; concatenate.

--------------------------------------------------
**Example: Base MLP Block and Model (Equinox)**

```python
import equinox as eqx
import jax, jax.numpy as jnp
from typing import Sequence
from dataclasses import dataclass

class MLPBlock(eqx.Module):
    ln: eqx.nn.LayerNorm
    in_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    dropout_rate: float
    key_dropout: bool = eqx.static_field()  # whether to use dropout

    def __call__(self, x, *, key, inference: bool):
        y = self.ln(x)
        y = self.in_proj(y)
        y = jax.nn.silu(y)
        if self.dropout_rate > 0 and not inference:
            # Explicit key per block invocation
            y = eqx.nn.Dropout(self.dropout_rate)(y, key=key)
        y = self.out_proj(y)
        return x + y

class MLP(eqx.Module):
    in_proj: eqx.nn.Linear
    blocks: list          # list[MLPBlock]
    ln_final: eqx.nn.LayerNorm
    out_proj: eqx.nn.Linear
    num_classes: int = eqx.static_field()

    def __call__(self, x, *, key, inference: bool = False):
        x = self.in_proj(x)
        keys = jax.random.split(key, len(self.blocks) + 1)
        for b, k in zip(self.blocks, keys[:-1]):
            x = b(x, key=k, inference=inference)
        x = self.ln_final(x)
        x = self.out_proj(x)
        return x
```

--------------------------------------------------
**Helper: Compute Stage Boundaries**

```python
def compute_stage_bounds(num_layers: int, num_stages: int):
    layers_per = (num_layers + num_stages - 1) // num_stages
    bounds = []
    for s in range(num_stages):
        start = s * layers_per
        end = min((s+1)*layers_per, num_layers)
        if start >= end:
            end = start  # empty stage (rare if layers < stages)
        bounds.append((start, end))
    return tuple(bounds)
```

--------------------------------------------------
**Option A: Simple Pipeline (Replicated Params)**

```python
def run_stage(x, module: MLP, stage_idx, stage_bounds, *, keys, inference: bool):
    # Create per-stage functions to avoid Python conditionals inside loop
    def make_fn(start, end):
        def f(x_):
            for blk, k in zip(module.blocks[start:end], keys[start:end]):
                x_ = blk(x_, key=k, inference=inference)
            return x_
        return f
    fns = tuple(make_fn(s, e) for s, e in stage_bounds)
    # stage_idx is a scalar; lax.switch picks correct function
    return jax.lax.switch(stage_idx, fns, x)

def pipeline_forward(module: MLP, x, *, key, num_microbatches: int, axis_name: str):
    num_stages = jax.lax.psum(1, axis_name)
    stage_idx = jax.lax.axis_index(axis_name)

    batch = x
    B = batch.shape[0]
    assert B % num_microbatches == 0
    mb = B // num_microbatches
    microbatches = batch.reshape(num_microbatches, mb, *batch.shape[1:])

    # Pad with dummy inputs for pipeline flush
    padded = jnp.concatenate(
        [microbatches,
         jnp.zeros((num_stages - 1, mb, *batch.shape[1:]), dtype=batch.dtype)],
        axis=0
    )
    stage_bounds = compute_stage_bounds(len(module.blocks), num_stages)
    # Pre-generate dropout keys per layer per iteration (simple version)
    # (You can fold in iteration index as well if needed)
    layer_keys = jax.random.split(key, len(module.blocks))

    hidden_dim = module.in_proj.out_features
    state0 = jnp.zeros((mb, hidden_dim), batch.dtype)

    def step(carry, inp):
        carry_state = carry
        stage_input = jnp.where(stage_idx == 0, inp, carry_state)
        y = run_stage(stage_input, module, stage_idx, stage_bounds,
                      keys=layer_keys, inference=False)
        last_idx = num_stages - 1
        emitted = jnp.where(stage_idx == last_idx, y, jnp.zeros_like(y))
        # ring-ppermute activation
        y_next = jax.lax.ppermute(
            y,
            axis_name,
            perm=[(i, (i+1) % num_stages) for i in range(num_stages)]
        )
        return y_next, emitted

    _, outs = jax.lax.scan(step, state0, padded)
    outs = outs[-num_microbatches:]  # discard warmup zeros
    return outs.reshape(B, hidden_dim)
```

Usage (with mesh + pjit/shard_map) simply calls `pipeline_forward` inside a jitted function. Memory downside: every device stores all layers.

--------------------------------------------------
**Option B: Stage-Stacked Parameters (Memory / Sharding Friendly)**

Idea: Add a leading dimension of size `num_stages` to each parameter; zero out slices for stages that do not use a layer; index by `axis_index` at call time:

```python
import equinox as eqx
import jax, jax.numpy as jnp

def stage_stack_module(module: MLP, stage_bounds, num_stages):
    # Determine which block indices each stage uses
    per_stage_sets = [set(range(s, e)) for s, e in stage_bounds]

    def annotate(tree):
        flat, treedef = jax.tree_util.tree_flatten(tree)
        new_flat = []
        for leaf in flat:
            if not isinstance(leaf, jax.Array):
                new_flat.append(leaf)
                continue
            # Heuristic: figure out if this leaf belongs to a particular block
            # For simplicity assume leaf size matches uniquely a block param; in practice
            # you’d pass path metadata or restructure blocks distinctly.
            variants = []
            for s in range(num_stages):
                # If stage uses the leaf -> real value; else zeros
                use_leaf = True  # refine with path metadata if desired
                variants.append(leaf if use_leaf else jnp.zeros_like(leaf))
            stacked = jnp.stack(variants, axis=0)  # (stages, *param_shape)
            new_flat.append(stacked)
        return jax.tree_util.tree_unflatten(treedef, new_flat)

    return annotate(module)

def index_stage(module_stacked, axis_name: str):
    stage_idx = jax.lax.axis_index(axis_name)
    def take(leaf):
        if isinstance(leaf, jax.Array) and leaf.ndim > 0 and leaf.shape[0] == jax.lax.psum(1, axis_name):
            return jax.lax.dynamic_index_in_dim(leaf, stage_idx, axis=0, keepdims=False)
        return leaf
    return jax.tree_map(take, module_stacked)
```

Then inside the forward you first “index” the module for this stage:

```python
def forward_stacked(module_stacked, x, key, *, num_microbatches, axis_name):
    local_module = index_stage(module_stacked, axis_name)
    # Reuse pipeline_forward logic but with a pruned module (only its stage’s block subset).
```

You would need reliable path metadata to zero out only the relevant layers; one pragmatic method is to store blocks as a list and perform stacking only for block parameters, leaving others (like in/out projections) to be masked separately (similar to Flax masking).

Sharding: Under `pjit` assign `PartitionSpec('model', ...)` to every stacked parameter so each device holds only its slice.

--------------------------------------------------
**Option D: Class-Level Decorator (Type Transformation)**

A decorator that returns a subclass injecting pipeline scheduling:

```python
import functools
import equinox as eqx
import jax, jax.numpy as jnp

def pipelineize(num_stages: int, num_microbatches: int,
                axis_name: str = "model",
                split_layers: callable | None = None):
    """
    Decorator turning an eqx.Module with attribute `blocks` (sequence)
    into a pipelined version.
    """
    def deco(BaseCls):
        class Pipelined(BaseCls):
            _num_microbatches: int = eqx.static_field()
            _axis_name: str = eqx.static_field()
            _stage_bounds: tuple = eqx.static_field()

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                L = len(self.blocks)
                if split_layers is not None:
                    self._stage_bounds = tuple(split_layers(L, num_stages))
                else:
                    self._stage_bounds = compute_stage_bounds(L, num_stages)
                self._num_microbatches = num_microbatches
                self._axis_name = axis_name

            def __call__(self, x, *, key, inference: bool = False):
                num_stages_local = jax.lax.psum(1, self._axis_name)
                stage_idx = jax.lax.axis_index(self._axis_name)
                B = x.shape[0]
                assert B % self._num_microbatches == 0
                mb = B // self._num_microbatches
                microbatches = x.reshape(self._num_microbatches, mb, *x.shape[1:])
                padded = jnp.concatenate(
                    [microbatches,
                     jnp.zeros((num_stages_local - 1, mb, *x.shape[1:]),
                               dtype=x.dtype)],
                    axis=0
                )
                layer_keys = jax.random.split(key, len(self.blocks) + padded.shape[0])  # simple key schedule
                stage_bounds = self._stage_bounds
                in_hidden = self.in_proj
                # Run input layer on stage 0 only (mask others)
                def apply_input(inp):
                    return jnp.where(stage_idx == 0,
                                     self.in_proj(inp),
                                     jnp.zeros((mb, self.in_proj.out_features), x.dtype))
                hidden0 = jnp.zeros((mb, self.in_proj.out_features), x.dtype)

                def stage_fn(x_stage, start, end, keys_block, inference):
                    y = x_stage
                    for blk, k in zip(self.blocks[start:end], keys_block):
                        y = blk(y, key=k, inference=inference)
                    return y

                def make_stage_func(start, end):
                    def f(z, keys_block):
                        return stage_fn(z, start, end, keys_block, inference)
                    return f
                stage_fns = tuple(make_stage_func(s, e) for s, e in stage_bounds)

                def scan_step(carry, scan_idx):
                    prev = carry
                    inp = padded[scan_idx]
                    stage_input = jnp.where(stage_idx == 0, inp, prev)
                    # Input dense once at first “real” microbatch entry per stage:
                    stage_input = jnp.where(
                        (stage_idx == 0),
                        stage_input,
                        stage_input
                    )
                    # Run only local blocks:
                    start, end = stage_bounds[stage_idx]
                    block_keys = layer_keys[:len(self.blocks)]  # naive reuse
                    y = jax.lax.switch(stage_idx, stage_fns, stage_input, block_keys[start:end])
                    last_idx = num_stages_local - 1
                    emitted = jnp.where(stage_idx == last_idx,
                                        y,
                                        jnp.zeros_like(y))
                    y_next = jax.lax.ppermute(
                        y,
                        self._axis_name,
                        perm=[(i, (i + 1) % num_stages_local) for i in range(num_stages_local)]
                    )
                    return y_next, emitted

                _, outs = jax.lax.scan(scan_step, hidden0,
                                       jnp.arange(padded.shape[0]))
                outs = outs[-self._num_microbatches:]
                final = outs.reshape(B, -1)
                # Output head only on last stage
                final = jnp.where(stage_idx == num_stages_local - 1,
                                  self.out_proj(self.ln_final(final)),
                                  jnp.zeros((B, self.out_proj.out_features), final.dtype))
                return final
        return Pipelined
    return deco
```

Usage:

```python
@pipelineize(num_stages=4, num_microbatches=8, axis_name="model")
class PipelineMLP(MLP):
    pass

model = PipelineMLP(
    in_proj=eqx.nn.Linear(in_features=784, out_features=512, key=jax.random.PRNGKey(0)),
    blocks=[MLPBlock(
        ln=eqx.nn.LayerNorm(512),
        in_proj=eqx.nn.Linear(512, 512*1, key=jax.random.PRNGKey(i*2+1)),
        out_proj=eqx.nn.Linear(512*1, 512, key=jax.random.PRNGKey(i*2+2)),
        dropout_rate=0.1,
        key_dropout=False
    ) for i in range(8)],
    ln_final=eqx.nn.LayerNorm(512),
    out_proj=eqx.nn.Linear(512, 10, key=jax.random.PRNGKey(999)),
    num_classes=10
)

# Mesh setup omitted; inside a pjit/shard_map:
def step(params, x, key):
    return jax.jit(lambda p, x_, k: p(x_, key=k))(model, x, key)
```

(You would typically separate params/static with `eqx.partition` before `pjit` if you want finer-grained sharding control.)

--------------------------------------------------
**Loss Masking & Metrics (Equinox)**

Equivalent to Flax loss masking:

```python
def loss_fn(model, params, x, y, key, *, axis_name="model"):
    logits = model(x, key=key, inference=False)
    stage_idx = jax.lax.axis_index(axis_name)
    last_idx = jax.lax.psum(1, axis_name) - 1
    loss_vec = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    loss_vec = jnp.where(stage_idx == last_idx, loss_vec, 0.0)
    correct = jnp.where(stage_idx == last_idx,
                        (jnp.argmax(logits, -1) == y),
                        False)
    loss = loss_vec.mean()
    metrics = {
        "loss": (loss_vec.sum(), jnp.where(stage_idx == last_idx, x.shape[0], 0)),
        "acc": (correct.sum(), jnp.where(stage_idx == last_idx, x.shape[0], 0))
    }
    return loss, metrics
```

Then `psum` metrics across (data, model) axes.

--------------------------------------------------
**Key Points When Porting to Equinox**

- Parameter sharding: Use a leading stage dimension + `PartitionSpec('model', ...)`, + `pjit`. You must supply sharding in/out specs explicitly (or rules).
- RNG folding: Implement `fold_rng_over_axis` manually: `jax.random.fold_in(key, jax.lax.axis_index(axis_name))`.
- Masking input/output: Conditional `where(stage_idx==0, ...)` or zeroed parameters for other stages.
- Avoid Python control flow on device-dependent values; use `lax.switch` / `lax.cond`.
- Microbatch scheduling: identical logic; ensure scan length = microbatches + (stages - 1).

--------------------------------------------------
**Mapping Flax Concepts to Equinox**

- `ModelParallelismWrapper` → Either (a) decorator adding stage dimension to params, or (b) function that stacks params + indexing helper.
- `stack_params` / `unstack_params` → Stage-stacking transform + runtime indexing (Option B).
- `PipelineModule` → Decorator-injected `__call__` performing scheduling (`pipelineize`).
- Parameter masking → Zero slices in stacked dimension or conditional application.

--------------------------------------------------
**Suggested Implementation Path**

1. Start with Option A (replicated) to validate pipeline scheduling.
2. Add masking of in/out layers (conditioned on `stage_idx`).
3. Introduce parameter stage stacking (Option B) and sharding via `pjit`.
4. Wrap whole thing with the decorator (Option D) to achieve type-level transformation.
5. Optimize RNG & key handling (unique per microbatch + stage).
6. (Optional) Add backward overlap schedules (1F1B) if needed later.

--------------------------------------------------
**Next Steps / Offers**

- I can provide a concrete `pjit` setup and parameter partition specs for the stage-stacked version if you share your mesh layout.
- I can adapt this to a Transformer-style module with attention blocks.
- I can extend the decorator to automatically stage-stack parameters instead of leaving that manual.

Let me know which of those you’d like to see next.
