We have enough sources to explain. Provide detailed explanation referencing code structure: initialization path, dynamic class rewriting, param group creation, hooking, communication operations (all-gather and reduce-scatter) including pre/post-forward, pre/post-backward operations; stream management and overlapping strategies; FSDPParam's roles in managing sharded/unsharded parameter states and type casting, extension points for pre/post all-gather transforms, handling gradient reduction dividing, hooking pre_backward etc.

Need to answer user's request: "tell me how pytorch implement FSDP2 in details, like how they add the all_gather and reduce_scatter hooks, and how they transforms module to fsdpmodule and how the init this new module etc"

Provide deep dive.

We should not run more commands; we already read relevant files.

Need to produce concise but thorough explanation.

Given instructions: For final answer with sections and bullets.

Should not include file line citations but can mention file paths and functions.

Should explain:

- Entrance API: fully_shard (torch.distributed.fsdp.fully_shard)
- Differences FSDP2 vs FSDP1 (conceptually).
- Transformation of module into dynamic subclass FSDP<OrigClass> by altering __class__ to new type inheriting (FSDPModule, orig cls) giving FSDPModule methods to user; override __new__ to circumvent for indexing into containers (like modulelist)? Actually inside FSDPModule.__new__ revert to original class for container operations? Need interpret:

In fully_shard(), after FSDPParamGroup creation, for each module in modules list, new_cls = type(f"FSDP{cls.__name__}", (FSDPModule, cls), dct) and module.__class__ is set to new dynamic class. FSDPModule defines __new__ to instantiate original underlying class when constructing new modules? Actually FSDPModule.__new__ is used when instantiating new dynamic class (FSDP... ), but for already constructed module this does not apply? Wait: When we create new class type("FSDP<Orig>", (FSDPModule, orig_cls), dct) and then assign module.__class__ = new_cls. We are not constructing new object; we are mutating the class pointer of existing instance. FSDPModule.__new__ will only matter when new instances of that dynamic subclass are created (e.g., copying or container operations). They override __new__ to skip layering FSDPModule wrappers and directly produce orig class object to avoid unintentional new FSDP-wrapped objects; they call orig_cls.__new__ and orig_cls.__init__. This ensures the dynamic subclass doesn't break methods that produce new modules (like __getitem__ indexing? Actually for container modules, retrieving an element returns stored module object, not constructed new. __new__ override matters if user constructs FSDP<ExistingType>. It prevents unnatural uses maybe. Provide explanation.

- fully_shard.state decorator: obtains per-module FSDPState via contract decorator; adds state into module states mapping.

- FSDPState.init registers forward pre and post hooks: if modules length ==1 uses register_forward_pre_hook and register_forward_hook; else group registration with multi-handle for multiple modules forming one FSDP param group.

- Pre-forward root hooking ensures lazy init: sets root, gather streams, ensures wait for optimizer event or current stream.

- Parameter grouping: _get_managed_modules collects modules for grouping; _get_managed_states collects parameters and buffers ignoring specified; _move_states_to_device moves them; then FSDPParamGroup created for list of params, modules.

- FSDPParamGroup sets up FSDPParam per param; each FSDPParam manages state transitions and collects all gather inputs; sharding logic in FSDPParam._init_sharded_param: create padded sharded parameter data chunk from original param; create DTensor with per-parameter sharding spec; replace module's parameter with sharded version via unsafe_setattr_param.

- FSDPParamGroup hooks executed in pre_forward and post_forward (FSDPParamGroup.pre_forward / post_forward) which in turn call unshard()/reshard().

- unshard triggers all-gather if currently sharded: calls foreach_all_gather collecting all gather inputs (fast path using torch._foreach_copy_ into single contiguous buffer, using custom torch.ops.fsdp.all_gather_copy_in to allocate output and flatten inputs). Manage asynchronous operations and overlapping via FSDPCommContext streams: all_gather_copy_in_stream and all_gather_stream. All_gather_result holds buffer, event, work object. Then wait_for_unshard() triggers foreach_all_gather_copy_out after collective completes, splits and (if needed) chunk-cat recomposing non-dim0 sharding, then sets unsharded param state.

- Overlap design: When not using async_op and in forward training state, they defer freeing last all_gather_result to allow next prefetch's copy-in to overlap with current copy-out (Note referencing [Note: Overlapping all-gather copy-in and all-gather]). They hold previous all_gather_result in comm_ctx.all_gather_state until next copy-out event recorded.

- Backward hooking: In FSDPState._post_forward they call _register_pre_backward_hook over outputs to register Tensor hooks that call FSDPState._pre_backward for each gradient flowing back. Pre-backward sets training state, maybe triggers prefetching for backward adjacency; FSDPParamGroup.pre_backward does unshard (if necessary) for backward; FSDPParamGroup.post_backward executed after autograd accumulates gradient for each parameter via RegisterPostBackwardFunction hooking into inputs that require grad when root config skip hooks is set or via root post backward final callback, else each parameter group post_backward collects grads, reduce-scatters them via foreach_reduce, then optionally all-reduce for HSDP replicate dimension, divides, casts back to original dtype, and writes into sharded_param.grad.

- The reduce-scatter operation uses custom chunk_cat and other ops; chunk-cat flattens unsharded grads into contiguous buffer with world_size chunking arrangement for reduce_scatter simultaneous; divides factors for gradient scaling for numeric stability; for HSDP extends to all-reduce stage.

- Overlap of compute and reduce-scatter: uses separate reduce_scatter_stream. After post_backward_reduce, they record events in stream to finalize.

- finalize_backward and root post backward final callback ensures pending operations are synchronized, state cleaned, and reshard/unshard events cleared. FSDPState._root_post_backward_final_callback runs at end of backward via ExecutionEngine queue_callback, ensures any param groups not used complete post_backward logic.

- Additional features: set_modules_to_forward_prefetch/backward to control explicit prefetch sequence; unshard_in_backward override; set_requires_gradient_sync for gradient accumulation.

- Mixed precision: FSDPParamGroup._init_mp_dtypes sets uniform dtype expectations; FSDPParam.init_dtype_attrs; param_dtype and reduce_dtype controlling pre-all-gather casting and reduction casting; foreach_all_gather uses _to_dtype_if_needed; foreach_reduce handles conversion and upcasting etc.

Need to mention hooking of custom all-reduce hook via FSDPModule.set_all_reduce_hook accepted and executed after reduce-scatter or after HSDP all-reduce.

Need to mention dynamic parameter registration/resets for operations like ._apply or load_state_dict.

Given complexity, produce structured explanation with sections:

- Overview & Goals
- Entry Points & Public API
- Module Transformation (dynamic subclassing, FSDPModule features)
- State Objects & Contract Mechanism
- Initialization Pipeline (fully_shard flow)
- Parameter Group Construction & Sharding (FSDPParamGroup, FSDPParam)
- Forward Path Hooks (pre-forward unshard, all-gather, overlap, post-forward reshard logic)
- Backward Path Hooks (registration, pre-backward unshard, post-backward gradient processing)
- Collective Implementations (all_gather, reduce_scatter, all_reduce details)
- Overlap & Stream Management
- Resharding Options (reshard_after_forward variants including integer world sizes)
- Mixed Precision & Offload
- Gradient Accumulation & Custom Hooks
- Lazy Init & Root Determination
- Torch.compile / Traceable FSDP2 invariants
- Extension Points (pre/post all-gather extension via tensor methods)
- Summary of Key Differences vs FSDP1 (brief)
- Key Files & Functions map

Answer must be direct, referencing code at high-level only, not citing line numbers; using file paths.

Let's craft answer.

Need to convey hooking mechanism: register_forward_pre_hook + register_forward_hook or group multi-handle; backward hooking via register_hook on outputs; additional autograd Function for post_backward hooking when torch._dynamo.config.skip_fsdp_hooks set.

Detailed explanation for FSDPParamGroup.unshard & wait_for_unshard; for reduce_scatter FSDPParamGroup.post_backward -> foreach_reduce.

Edge cases: even sharding required on non-dim0 while using Shard(dim) not zero.

We can also mention how FSDPParam.all_gather_inputs computed; optimization using foreach copy to flatten and copy multiple parameters at once; all_gather_copy_in custom op; splitting/fanning out with split_with_sizes_copy; chunk-cat for reduce-scatter input.

Finally, mention dynamic device selection, ignoring modules, etc.

Let's produce answer.

Ensure disclaimers: Implementation details may evolve. Provide suggestions to trace code: set environment variable for debug logs.

Ok.

Return answer.

No plan tool needed because user just requested explanation; we are done.

Provide answer.


**Overview**
- **Goal:** FSDP2 (aka fully_shard) provides per-parameter dim‑0 sharding using DTensor, grouping parameters into communication units (FSDPParamGroup) to perform a single all_gather before use and a single reduce_scatter (plus optional all_reduce for HSDP) after backward, with explicit/implicit prefetch and careful stream overlap.
- **Key Idea:** Instead of flattening like FSDP1, FSDP2 shards each original parameter directly (DTensor when possible), dynamically swaps the module’s class to inject FSDP control methods, and manages sharded/unsharded transitions via storage resizing and view rewiring.

**Entry API**
- **`fully_shard(module, ...)`** (torch/distributed/fsdp/_fully_shard/_fully_shard.py): Applies FSDP2 in place; returns the same module (or list) after instrumentation.
- **`FSDPModule` methods** (same file): Runtime control (`unshard`, `reshard`, prefetch setters, sync toggles, custom hooks).
- **`register_fsdp_forward_method`**: Wraps an arbitrary forward-like method with FSDP pre/post hooks.

**Module Transformation**
- In `fully_shard`, each managed module’s `__class__` is replaced with a dynamically created subclass `FSDP<OrigClass>` that inherits `(FSDPModule, OrigClass)`.
- `FSDPModule.__new__` returns an instance of the original class for any future instantiation, preventing accidental recursive wrapping.
- This gives every wrapped module FSDP control APIs without adding wrapper container objects (preserves existing references).

**State Objects & Contract**
- Decorator `@contract(state_cls=FSDPState)` attaches a 1:1 `FSDPState` to the (root) module (torch/distributed/fsdp/_fully_shard/_fully_shard.py + _fsdp_state.py).
- `FSDPState` stores: param group (if this call grouped parameters), forward/backward training state, prefetch targets, shared communication context, root/lazy-init metadata.

**Initialization Flow (fully_shard)**
1. Normalize modules (single or list), build or validate `DeviceMesh`.
2. Decide post-forward mesh shape via `_get_post_forward_mesh_info` (handles bool/int `reshard_after_forward`).
3. Gather “managed modules” bottom-up (excluding already-sharded or ineligible) via `_get_managed_modules`.
4. Collect parameters/buffers (excluding ignored) via `_get_managed_states`.
5. Move them to device (`_move_states_to_device`).
6. Create `FSDPParamGroup` if there are parameters; inside it create one `FSDPParam` per parameter (shards + metadata).
7. Mark modules with internal flags used by Dynamo (`_is_fsdp_managed_module`, `_fsdp_use_orig_params`).
8. Dynamic subclass swap (adds FSDPModule methods).
9. (Lazy) root determination deferred until first forward (`_lazy_init`).

**Parameter Group Construction**
- **`FSDPParamGroup`** (_fsdp_param_group.py): Holds ordered list of `FSDPParam`, modules it covers, mixed precision & offload policy, communication streams, reduction policy flags, mesh info, hook bookkeeping.
- Groups are deliberately *non-flattened*; each param remains distinct allowing per-param extensions.
- One param group ↔ one pair of collectives per iteration (one all_gather, one reduce_scatter (+ optional all_reduce)).

**Per-Parameter Management (FSDPParam)**
- Shards original parameter: splits along chosen (default dim 0) with padding for uneven last shard (if dim=0) and builds a padded local 1D storage view.
- Wraps local shard in DTensor spec (with possible composite placements for HSDP or TP) and replaces the module’s reference via `unsafe_setattr_param`.
- Tracks three sharded states: SHARDED, SHARDED_POST_FORWARD (resharded to smaller world size), UNSHARDED.
- Reconstructs unsharded parameter after all_gather using views (or post-processing extension hook).
- Mixed precision dtype attributes (`param_dtype`, `reduce_dtype`) initialized lazily.

**Forward Hooking**
- `FSDPState.init` registers:
  - If single module: standard `register_forward_pre_hook` / `register_forward_hook`.
  - If multiple modules grouped: a composite mechanism (`_register_group_forward_hooks`) where the first module triggers pre, the last triggers post.
- **Pre-forward (`FSDPState._pre_forward`):**
  - Runs root lazy init once (determines root, assigns FQNs, shares comm context, finalizes param-group lazy inits).
  - Casts inputs if `mp_policy.cast_forward_inputs`.
  - Delegates to `FSDPParamGroup.pre_forward` which:
    - Sets training state, triggers `unshard` (possibly async), waits for copy-out completion (`wait_for_unshard`), sets post-backward autograd hook registration for gradient accumulation (via a custom autograd Function if hooks skipped).
    - Issues implicit forward prefetch of next param groups (overlap).
- **Post-forward (`FSDPState._post_forward`):**
  - `FSDPParamGroup.post_forward` reshards (if configured) or keeps unsharded.
  - Registers pre-backward hooks on outputs (tensor-level) for backward-time unshard.
  - Manages freeing last deferred all_gather output if necessary.
  - Casts outputs to `output_dtype` if specified.

**All-Gather Implementation**
- Triggered in `FSDPParamGroup.unshard`.
- Uses two streams (copy-in + collective) from `FSDPCommContext` when overlap enabled (async_op=False in forward/backward pre-phase); otherwise current stream.
- `foreach_all_gather` (_fsdp_collectives.py):
  - Builds flattened list of all-gather inputs across params (fast path: grouped foreach copy into interleaved flat buffer).
  - Custom op `fsdp.all_gather_copy_in` packs inputs contiguously (records per-param split sizes & dtypes).
  - Issues `dist.all_gather_into_tensor` producing a single flat output.
- `wait_for_unshard`:
  - `foreach_all_gather_copy_out` splits flat output back into per-param outputs using custom op `fsdp.split_with_sizes_copy`.
  - Handles non-dim0 sharding by temporary reorder then chunk-cat back to correct dimension.
  - Initializes unsharded parameter views (and runs post-all-gather transform if extensions present).
  - Defers freeing previous gather buffer for forward overlap (Note: Overlapping all-gather copy-in and all-gather).
- Mixed precision: Inputs optionally cast beforehand; unsharded parameter constructed in `param_dtype` (or original dtype if no cast).

**Backward Hook Path**
- Tensor gradient hooks (from post-forward) call `FSDPState._pre_backward`:
  - Registers root final callback, triggers param group `pre_backward`: unshard (if resharded after forward or not persisted) and optional backward prefetch (reverse post-forward order) unless explicit list set.
- Gradient accumulation logic:
  - `post_backward` (per param group) collects and strips gradients from unsharded params, reshard (if configured), then launches reduce_scatter (and optional all_reduce).
  - If gradients are of higher precision than `reduce_dtype`, downcasts before copying into flat reduce_scatter input.
- Root final callback (`_root_post_backward_final_callback`):
  - Ensures every group runs `post_backward` even if its forward inputs had no grad path.
  - Finalizes states (optionally frees buffers if last microbatch).
  - Synchronizes outstanding reduce_scatter or all_reduce events.

**Reduce-Scatter + (Optional) All-Reduce**
- Implemented in `foreach_reduce` (_fsdp_collectives.py):
  - Preprocess: For non-dim0 shard dims, re-chunk and re-concatenate grads so reduce_scatter expects dim0 segmentation.
  - Packs (chunk_cat custom op) all grads into one flat tensor laid out for reduce_scatter (world_size rows).
  - Optionally applies pre-division (for fp16 stability) using factor ~√N; uses NCCL PreMulSum if custom user divide factor set.
  - Calls `dist.reduce_scatter_tensor` with AVG (or SUM + manual divide) producing local shard slice.
  - HSDP path: accumulate partial reduce outputs across shard mesh, then all_reduce across replicate mesh (with AVG or SUM). Stores event & tensor for deferred synchronization and potential casting.
  - User hook (`_all_reduce_hook`) may run after reduce_scatter or after HSDP all_reduce on its own (possibly provided stream).
  - Post divide + cast back to original dtype; scatter results are written into sharded param grads either by accumulation or DTensor construction.

**Overlap & Streams**
- Streams: `all_gather_copy_in_stream`, `all_gather_stream`, `reduce_scatter_stream`, `all_reduce_stream`.
- Forward overlap: next all_gather copy-in overlaps with current all_gather collective; current copy-out overlaps with subsequent compute; deferred freeing ensures memory reuse safety.
- Backward overlap: reduce_scatter runs concurrently while earlier parameter gradients still computing; all_reduce (HSDP) runs concurrently with later reduce_scatter operations.
- Events tie lifetimes between streams; final callback ensures default stream waits on reduction completion before optimizer.

**Resharding Controls**
- `reshard_after_forward`:
  - True: always free unsharded params post forward (min memory).
  - False: keep unsharded until after backward (saves one all_gather).
  - Int divisor: reshard to a *smaller* world size (intermediate replication) for a trade-off (stores reduced subset world-size slice).
- Post-forward mesh logic implemented via `_get_post_forward_mesh_info`; reshard uses separate state `SHARDED_POST_FORWARD`.

**Mixed Precision & Offload**
- `MixedPrecisionPolicy`: independent param/reduce/output dtypes; enforced uniform per group.
- Grad reduction can be at lower precision (`reduce_dtype`) with deferred upcast if accumulating.
- `CPUOffloadPolicy`: Parameters allocated/padded on CPU; D2H/H2D copies orchestrated with pinned memory for overlap; gradient copy-out events tracked per param.

**Gradient Accumulation / No Sync**
- `set_requires_gradient_sync(False)`: skip reduce_scatter/all_reduce; grads retained unsharded or converted to accumulated grads; later re-enabled sync will include them.
- `set_requires_all_reduce(False)` for HSDP: keep partial reduce (reduce_scatter only).
- `set_reshard_after_backward(False)`: keep unsharded params across accumulation steps to save future all_gathers.

**Custom All-Reduce Hook**
- `set_all_reduce_hook` sets user post-reduction callback (e.g. quantization, compression) executed in chosen stream (except native HSDP always uses internal stream).

**Lazy Root Determination**
- First forward through the (true) root triggers `_lazy_init`: marks root, enumerates all subordinate FSDP states, sets their `_is_root`, initializes shared comm context & FQNs, and flips root's `post_forward_mesh_info` to None if auto policy to avoid redundant forward-backward all_gather cycle.

**Torch.compile / Traceable FSDP2**
- Special invariants (see notes in _fsdp_param.py):
  - Recreate `all_gather_outputs` each time (avoid graph inputs).
  - Always copy into persistent `nn.Parameter` buffer (resize_(full)->copy_->resize_(0) pattern possibly eliminated by an Inductor pass).
  - Avoid functionalization for `fsdp.copy_` and `resize_storage_bytes_` to keep mutation semantics predictable inside compiled graphs.
  - RegisterPostBackwardFunction bypassed under traceable mode; `pre_backward` triggers pending `post_backward`.

**Extension Points**
- Inner tensor may define `fsdp_pre_all_gather` and `fsdp_post_all_gather` to transform inputs/outputs (e.g., quantization, low-rank packing).
- Metadata and multi-tensor support handled via `ExtensionsData` (saved sizes, metadata transfer).

**Differences vs FSDP1**
- Per-parameter sharding (no giant flat parameter) => simpler state dict handling and better composability with DTensor/TP.
- Explicit param group object controlling collectives & overlap.
- Streamed overlap with dual-stream all_gather copy-in/copy-out separation.
- Simpler reshard semantics (boolean or integer partial replication).
- Direct module class mutation instead of wrapper object (lower indirection).
- Built-in flexible extension hooks for all_gather transforms.

**Key Files / Functions**
- API & module transform: `torch/distributed/fsdp/_fully_shard/_fully_shard.py` (`fully_shard`, `FSDPModule`)
- State & hooks: `.../_fsdp_state.py` (`FSDPState`, forward/backward hook logic)
- Param group: `.../_fsdp_param_group.py` (unshard/reshard, overlap, pre/post-backward)
- Param management: `.../_fsdp_param.py` (sharding, dtype casting, extension hooks)
- Collectives: `.../_fsdp_collectives.py` (`foreach_all_gather`, `foreach_reduce`, custom library ops)
- Initialization helpers: `.../_fsdp_init.py`
- Common utilities / enums: `.../_fsdp_common.py`

**Practical Tracing Tips**
- Enable debug logging via `logger = logging.getLogger("torch.distributed.fsdp.fully_shard"); logger.setLevel(logging.DEBUG)` to see hook ordering.
- Insert `record_function` markers already present (view in profiler) to observe overlap: `FSDP::all_gather`, `FSDP::all_gather_copy_out`, `FSDP::post_backward_reduce`, etc.

**Summary**
FSDP2 restructures FSDP around lightweight per-parameter sharding with an explicit param group abstraction, dynamic module class augmentation, carefully layered pre/post forward & backward hooks, and fused multi-parameter collectives implemented via custom ops plus stream-level overlap. The lifecycle is: SHARDED → (pre-forward all_gather) → UNSHARDED → (optional post-forward reshard) → (pre-backward unshard if needed) → reduce_scatter(+all_reduce) → SHARDED. Extensibility (pre/post all_gather transforms), mixed precision, offload, partial replication, and compile-time invariants are first-class concerns built into the design.

Let me know if you’d like a diagram, a walk-through on a tiny model, or a comparison of memory timelines vs FSDP1.
