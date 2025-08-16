Given user request, I'll propose a design for eqx.map_variables:

Goal:
- Accept a Module subclass or callable (like eqx.nn.Linear) and one or more mapping functions to transform parameter arrays inside the module instance at call time or initialization.
- Equivalent to Flax's map_variables for weight tying or weight transformations; For Equinox, weight tying can be achieved by customizing parameter initialization and referencing other weights.

But direct modifications after initialization likely require dynamic view of arrays; in Equinox arrays are stored as leaves; if we map them before call we just pass a transformed Module into call. simplest design:

def map_module(ModuleClass: Type[eqx.Module], map_in: Callable[[PyTree], PyTree], map_out: Optional[Callable[[PyTree], PyTree]] = None, mutate: bool = False, filter_spec = eqx.is_array) -> Type[eqx.Module]:

It returns a new dataclass subclass deriving from ModuleClass with overridden __call__ (and maybe other methods specified) such that before invoking super().__call__ it applies map_in to subset of leaves filtered, and after call if mutate is true, merges map_out result into self.

Complication: because Module is frozen dataclass, we cannot mutate fields; we'd need to produce modified copy perhaps using jax.tree_util.tree_map with sentinel, or equinox.tree_at.

Simpler is to treat mapping as ephemeral: inside call we apply transform on the Module instance (pure functional) and call underlying function with transformed weights; we don't update external state (which matches typical usage of masked weights). For weight normalization or standardization that must happen at initialization maybe—they use init=True path.

In Equinox no dedicated initialization procedure customizing mapping.

So replicating all features of flax.map_variables (init True to run mapping during initialization; support for mutability or hooking both mapping in/out) may require hooking into underlying function passes by eqx.filter_jit? But Equinox initialization is just constructing Module; you pass random key to module's constructor; there is no global scope variable that would differentiate parameter vs state. Unless we embed transform inside __init__? Provided modules in Equinox already handle initialization inside __init__ using provided key to random functions.

Therefore propose bridging use-case: Provide eqx.nn.map_parameters for modules; operations used: weight tying, parameter transformations, bit-level modifications.

We can maybe leverage function-level transform similar to filter_jit; create a wrapper Module class _MapModule with attributes _orig (callable or Module), _map_in, _map_out.

If orig is a function with signature f(*args, **kwargs). If used inside module's setup stage they'd do:

DenseMasked = eqx.map_parameters(eqx.nn.Linear, map_in=..., map_out=..., mutate=False)

Then they can instantiate DenseMasked(...)

Implement eqx.map_parameters such that if passed a Module subclass we produce subclass customizing __init__ call chain?

Simplify:

Case 1: User passes Callable that returns a Module instance when called, like eqx.nn.Linear(in_size, out_size, key=...). They want eqx.map_variables(nn.Dense, 'params', sign, init=True) style which uses a module class (nn.Dense) producing new Module instance when invoked with arguments.

In Equinox eqx.nn.Linear is a class; calling eqx.nn.Linear(in_size, out_size, key=key) returns Module instance. So we can implement:

def map_params(ModuleClass: Type[M], map_fn: Callable[[Any], Any], where: FilterSpec = eqx.is_array, reverse_fn: Optional[callable]=None, init: bool=False, mutate: bool=False):

Return new class:

class _MappedModule(ModuleClass):
    def __call__(self, *args, **kwargs):
        mapped_self = jtu.tree_map(lambda leaf: map_fn(leaf) if predicate else leaf, self)
        return super(_MappedModule, mapped_self).__call__(*args, **kwargs)

But in Python we cannot change 'self' argument passed to superclass __call__ easily. Instead we might override __call__:

def __call__(self, *args, **kwargs):
    mapped_module = eqx.tree_at(lambda x: x, self, ???)  # create new instance with mapped param leaves
    return ModuleClass.__call__(mapped_module, *args, **kwargs)

Mapping leaf selection: use jtu.tree_map with eqx.is_array to select arrays.

Define _map_leaves(pytree): jtu.tree_map(lambda x: map_fn(x) if predicate(x) else x, pytree)

But we must ensure that non-parameter attributes like activation functions not transformed.

Return value: Use functional transform; we don't update original instance, so repeated calls apply mapping again. Accept map_out to modify outputs perhaps.

Should we provide ability to transform only specific named leaves? In Flax map_variables finds variables in 'params' collection only; Equinox has no collections; we could require an is_leaf predicate or filter_spec, default eqx.is_array (like general arrays). Provide an API: filter_spec is PyTree with bool or callables, or we just accept a predicate function with signature leaf -> bool.

Simplify by reusing eqx.filter to produce a mask.

However underlying code design in Equinox uses filter_jit etc; new transform should integrate with module_update_wrapper to supply __wrapped__ property.

Simpler: _MapModule eqx.Module with fields: _module_class, _map_in, _map_out, _predicate, maybe _methods to intercept.

But if user wants to map only parameters of a specific module each time they instantiate. Approach:

def map_module(cls: Type[eqx.Module], map_in_fn, map_out_fn=None, where=eqx.is_array, methods=("__call__",)) -> Type[eqx.Module]:

Return new subclass implementing those methods:

def mapped_method(self, *args, **kwargs):
    mapped_self = jtu.tree_map(lambda leaf: map_in_fn(leaf) if predicate(leaf) else leaf, self)
    out = original_method(mapped_self, *args, **kwargs)
    if map_out_fn is not None:
        mapped_self_out = jtu.tree_map(lambda leaf: map_out_fn(leaf) if predicate(leaf) else leaf, mapped_self)
        # mutation? If mutate True, return new module object maybe?
    return out

Issue: original method expects 'self' of subclass; we provide mapped_self which is an instance of subclass? we'd produce new instance of same subclass with updated leaves maybe:

mapped_self = eqx.tree_at(...). The new instance will require type; we can use dataclasses.replace? modules are dataclasses @dataclass(frozen=True); rewriting leaves need customizing of entire tree; jtu.tree_map returns new dataclass automatically because dataclass is PyTree; JAX flatten/unflatten uses eqx internal flatten; transforms leaves replaced maintain type.

So mapping function returns a new subclass instance; all good.

Edge cases: For property other than __call__ maybe e.g., user wants mapping apply to encode method etc; Provide parameter 'methods' list of method names to wrap.

Attending user requirement: They say "i want the inputs are the type[eq.module] instead of just eq.moudule" meaning they want to accept class not instance.

Therefore propose to implement map_parameters for Equinox similar to flax.map_variables semantics but simplified for lack of variable collections; Accept Module class; returns subclass with transformed methods.

Need to implement file maybe equinox/_map.py exporting map_modules or map_parameters; update __init__.py.

But user hasn't asked to actually implement code yet, they ask "read flax_research directory, can you tell me how can i implement nn.map_variables to equinox counterpart ..."

So they want guidance on how, not necessarily actual code modifications. Provide explanation plus design.

We should still maybe look at Equinox patterns for gradient wrappers; those wrappers implement __get__ descriptor to handle being placed inside other modules; for class-level method modifications we just need standard function wrappers when assigned to class.

Return type: For usage inside module's __init__ or attribute assignment they want to do:

MaskedLinear = eqx.map_parameters(eqx.nn.Linear, map_in=..., map_out=..., mutate=False)
self.layer = MaskedLinear(in_size, out_size, key=...)

Similar pattern.

Hence map_parameters returns new subclass that inherits eqx.nn.Linear.

Alternative approach: Use a class decorator to produce dynamic subclass customizing __call__.

Key design decisions:

1) Provide map_in_fn and map_out_fn, default identity.
2) Provide filter_spec or predicate to restrict which leaves to transform. For arrays we detect with eqx.is_array.
3) Provide mutate bool: if True, apply map_out_fn to produce new arrays stored in module; for immutability we need to return new module object, but we cannot update self; users would capture returned new Module maybe by using pattern:

output, new_module = eqx.map_variables(fn, mutate=True)(module, *args)
But they desire the mapping to be hidden inside class; perhaps we choose to not support mutate (no variable state). Equivalent of Flax's init/mutable operations might be overkill.

Given user mention of 'we'll do something similar like filter_{jit, value_and_grad, etc}', that suggests they want a transform function that returns a wrapper Module (like filter_jit) and can be applied to functions or methods; mapping is ephemeral; Provide descriptor semantics.

Hence design variant #2:

Implement eqx.map_variables(fun, map_in_fn=..., map_out_fn=..., filter_spec= eqx.is_array, mutate=False). Accept fun: function expecting maybe some arguments; map transform will transform its first argument's Module leaves? Wait filter_jit expects a function f(*args); mapping variables refer to the 'variables' belonging to Module parameter state inside Flax's scope; analog in Equinox is the first argument if it is a Module instance. So design:

@eqx.map_module(map_fn) # returns wrapper for function expecting module first arg

Possible call pattern inside class method:

class C(eqx.Module):
    ...
    def __call__(self, x):
        return eqx.map_variables(eqx.nn.Linear, map_in_fn=mask)(...)? Hmm.

Better to accept Module class to align.

Therefore integration difference likely leaning on class-subclass approach.

But we should propose two complementary APIs:

1) map_module(cls, map_in, map_out=None, where=eqx.is_array) -> new class

2) map_module_call(map_in,...)(module_instance, *args) returning function akin to functional transform for existing module instance.

Simplify respond:

Will outline step-by-step path referencing relevant Equinox components:

- Use filter partition combine to separate leaves to transform
- Wrap using Module subclass or wrapper Module object with __call__ property, using module_update_wrapper for introspection
- Provide descriptor semantics so can assign to class

Edge-case: Accepting Type[Module] annotate.

Implement code skeleton.

Return recommended code snippet.

Need highlight differences from Flax:

- No separate variable collections (params vs batch_stats). Provide optional predicate or atr path filter.

- No init/mutable semantics; mapping always pure; if need to permanently update parameters user calls eqx.tree_at or eqx.apply_updates.

- RNG handing manual; if mapping trans_in_fn requires randomness, user passes key in; typical customizing uses deterministic transform.

Add optional in-place rewriting using tree_at if mutate True produce new Module; design an API returning wrapper with new module perhaps returning (output, new_module) pattern or customizing to replicate out variable change (like eqx.filter_value_and_grad returning gradient).

Simplify possible mutate semantics:

class _MapWrapper(eqx.Module):
    _fun: Callable
    _map_in: Callable
    _map_out: Callable
    _where: FilterSpec
    _mutate: bool

    def __call__(self, module: eqx.Module, *args, **kwargs):
         diff, static = partition(module, self._where)
         diff_mapped = jtu.tree_map(self._map_in, diff)
         mapped_module = combine(diff_mapped, static)

         out = self._fun(mapped_module, *args, **kwargs)
         if self._mutate:
             diff_after = jtu.tree_map(self._map_out, diff_mapped)
             updated_module = combine(diff_after, static)
             return out, updated_module
         else:
             return out

This looks analogous to map_variables over an instance. Accept module type maybe not needed; but using Type[Module] they want to map for all instances created from a subclass? Equivalent to convert a Module class to new Module class whose __call__ method first maps its parameters before calling original. Provide class-level transform:

def map_module_class(cls, map_in, map_out=id, mutate=False, where=is_array):

    orig_call = cls.__call__

    @functools.wraps(orig_call)
    def new_call(self, *args, **kwargs):
        diff, static = partition(self, where)
        diff_mapped = jtu.tree_map(map_in, diff)
        mapped_self = combine(diff_mapped, static)
        out = orig_call(mapped_self, *args, **kwargs)
        if mutate:
            diff_after = jtu.tree_map(map_out, diff_mapped)
            updated_self = combine(diff_after, static)
            # assign? can't assign bc frozen; just return out. To persist, user must assign manually
        return out
    new_cls = type(f"{cls.__name__}Mapped", (cls,), {"__call__": new_call})
    return new_cls

Return new class.

But for detect of 'where' we used eqx.is_array for diff, so we map all arrays (like parameters). Provide optional filter to restrict e.g. to only kernel or weight arrays by lambda path.

Path-based mapping ideally supply leaf paths; jtu.tree_map can't easily provide path; we can incorporate jax.tree_util.tree_map_with_path (exists maybe in JAX ready) or restful to supply path to map_in/out functions; So Option: Provide map_in/out functions of signature (leaf)->leaf or optional path parameter.

Simplify initial implement.

Edge Cases:

- ensure new class is also eqx.Module: it's subclass of original which is dataclass eqx.Module; Python dynamic subclass inherits dataclass; customizing dataclass metadata may not be necessary; dataclass decorators not executed for new subclass; but no new fields added so fine.

Returning new class accessible by Type[Module].

Now we need to test; but user asked for explanation; they did not require code editing.

Therefore respond with comprehensive plan referencing relevant code lines for impetus (but not quoting with file path referencing disclaimers—they said no citation style—they want explanation). Provide code skeleton.

We must not produce lines with bold markup maybe use bullet lists with guidelines. Provide sections maybe.

Given "read the flax_research directory" we have done reading; summarizing internal workings of flax.map_variables and providing adaptation.

Will mention differences, tradeoffs, sedation.

If they want hooking like treat mapping only during init vs call, mention at design.

Add path of new file would be perhaps equinox/_map.py or equinox/_transforms.py referencing internal pattern.

Also need integrate into __init__.py for export.

Implementation details referencing partitions.

Important to adapt descriptor semantics when used inside Module definitions:

class C(eqx.Module):
    layer: eqx.nn.Linear
    def __init__(...):
        MappedLinear = eqx.map_module(eqx.nn.Linear, map_in_fn=mask)
        self.layer = MappedLinear(in_size, out_size, key=key)
    def __call__(self, x):
        return self.layer(x)

Alternative usage for function-level mapping:

@eqx.map_variables(map_in_fn=mask)
def f(model, x):
    return model(x)

But user wants Type[eq.Module] input, so show that.

Will include error disclaimers.

Ok craft answer.

Need to follow final answer formatting guidelines.

So answer in final channel, no tool calls since they only asked conceptual design. Provide sections:

**Summary** maybe; **Flax map_variables mechanics**; **Equinox differences**; **Target API**; **Implementation Steps**; **Code Skeleton**; **Example Usage**; **Extensions**; **Caveats**; **Next Steps**

Bullet style.

Yes.

Proceed.


**Goal**  
- Provide an Equinox analogue of `flax.linen.map_variables` that (a) lets you transparently transform selected parameter leaves before (and optionally after) a module method runs, and (b) accepts a `Type[eqx.Module]` (a module class) – so you can do the Flax-style pattern: `MaskedLinear = eqx.map_variables(eqx.nn.Linear, map_in=..., map_out=...)` then instantiate it later.

**What Flax `map_variables` Actually Does**  
- Operates at the scope/collection layer: chooses variable collections (`'params'`, `'batch_stats'`, etc.).  
- Builds a “view” of chosen collections (`map_in_fn`), runs the target, then (optionally) writes back with `map_out_fn` if `init` or `mutable`.  
- Ensures: correct init semantics, controlled mutability of mapped collections, RNG safety, no double‑init of other state.  
- Can be applied to a Module class (e.g. wrapping `nn.Dense`) and later instantiated.

**Key Differences in Equinox**  
- No variable collections; parameters/state are just PyTree leaves of the module.  
- No mutation during a call (modules are frozen dataclasses); any “write back” must produce a new module value.  
- No scope/RNG layering to manage. Simpler: just partition leaves into “to map” vs “unchanged”.  
- Existing transforms (`filter_jit`, `filter_value_and_grad`, `filter_vmap`, etc.) return wrapper Modules (descriptor-compatible) when given a function. We can mirror that style.

**Proposed API (Class-Focused)**  
- `eqx.map_variables(cls, map_in_fn, map_out_fn=None, *, where=eqx.is_array, mutate=False, methods=("__call__",)) -> Type[eqx.Module]`  
  - `cls`: a `Type[eqx.Module]` to wrap.  
  - `map_in_fn(leaf)` (optionally path-aware later) transforms chosen leaves before each wrapped method call.  
  - `map_out_fn(leaf)` (optional) produces post-call leaves if `mutate=True` (e.g. reversible transforms, quantisation).  
  - `where`: filter spec / predicate deciding which leaves are “variables” (default: all array leaves). Could accept:
    - Callable `leaf -> bool`
    - PyTree of bool/callables prefixing the module (mirroring `eqx.filter` style)
  - `mutate=False`: if `False`, transformation is ephemeral (like masking). If `True`, returns a tuple `(output, updated_module)` or (for class wrapping) updates internally by producing a new frozen instance (user must capture it; cannot in-place assign).  
  - `methods`: iterable of method names to wrap (default only `__call__`).

**Alternate Functional API (Instance-Focused)**  
- `eqx.map_variables_fn(fun, map_in_fn, map_out_fn=None, *, where=eqx.is_array, mutate=False)`  
  - Returns a wrapper Module (like `filter_jit`) expecting the first argument to be a Module instance; partitions/transforms it and calls `fun`.

(You can implement class API first; functional form can reuse core helper.)

**Implementation Steps**  
1. Filtering:
   - Reuse `partition(module, where)` to split into `(to_map, static)`.
2. Mapping in:
   - `mapped_leaves = jtu.tree_map(map_in_fn, to_map)`
   - `mapped_module = combine(mapped_leaves, static)`
3. Call underlying method (`orig_method(mapped_module, *args, **kwargs)`).
4. Mapping out (if `mutate` and `map_out_fn` provided):
   - `post_leaves = jtu.tree_map(map_out_fn, mapped_leaves)`
   - `updated_module = combine(post_leaves, static)`
   - Return `(out, updated_module)` (document this).
5. Class wrapper:
   - Dynamically create subclass: `New = type(f"{cls.__name__}Mapped", (cls,), {...})`
   - For each method in `methods`, grab unbound function `orig = getattr(cls, name)`; define a closure implementing steps 1–4.
   - Preserve metadata: copy `__doc__`, `__module__`, annotations; optional: provide `__wrapped__` reference. (Since this is a full class, `module_update_wrapper` isn’t strictly needed unless you want consistency; you can also define a tiny wrapper Module instead of a subclass if you prefer.)
6. Edge: `mutate=True` inside class method cannot actually “rebind” `self`; document user must capture returned module if you choose to return it, or forbid mutate in class API initially.
7. Export:
   - Put code in new `equinox/_map.py`.
   - Add `from ._map import map_variables as map_variables` to `__init__.py`.
8. Tests:
   - Masking test (e.g. causal mask via upper-triangular).
   - Bit sign example (like Flax bit weights): map_in applies `jnp.sign`; confirm determinism.
   - Weight tie / transpose: wrap `eqx.nn.Linear` with `map_in_fn=lambda w: w.T` for selected leaves.
   - Ensure interaction with `filter_jit` (compose order both ways).
9. Documentation:
   - Brief docstring clarifying differences vs Flax (no collections; pure; optional mutate returns new module).

**Code Skeleton (Core Helper)**  
```python
# equinox/_map.py
from __future__ import annotations
import types
import functools as ft
import jax.tree_util as jtu
import equinox as eqx
from ._filters import partition, combine, is_array
from ._module import Module

def map_variables(
    cls,
    map_in_fn,
    map_out_fn=None,
    *,
    where=is_array,
    mutate=False,
    methods=("__call__",),
):
    if not isinstance(cls, type) or not issubclass(cls, Module):
        raise TypeError("First argument must be a subclass of eqx.Module.")
    if mutate and map_out_fn is None:
        map_out_fn = lambda x: x  # default identity on writeback

    def _wrap_method(name, orig_fn):
        @ft.wraps(orig_fn)
        def wrapped(self, *args, **kwargs):
            to_map, static = partition(self, where)
            mapped = jtu.tree_map(map_in_fn, to_map)
            mapped_self = combine(mapped, static)
            out = orig_fn(mapped_self, *args, **kwargs)
            if mutate:
                mapped_after = jtu.tree_map(map_out_fn, mapped)
                updated_self = combine(mapped_after, static)
                # No in-place mutation possible; return tuple
                return out, updated_self
            return out
        return wrapped

    dct = {}
    for m in methods:
        if not hasattr(cls, m):
            raise AttributeError(f"{cls.__name__} has no method {m}")
        orig = getattr(cls, m)
        # unwrap descriptor to function if needed
        if isinstance(orig, (types.FunctionType, types.MethodType)):
            func = orig.__func__ if isinstance(orig, types.MethodType) else orig
        else:
            raise TypeError(f"Method {m} is not a plain function; cannot wrap cleanly.")
        dct[m] = _wrap_method(m, func)

    New = type(f"{cls.__name__}Mapped", (cls,), dct)
    New.__doc__ = (cls.__doc__ or "") + "\n\n[Wrapped by eqx.map_variables]"
    return New
```

**Example Usage**  
- Weight masking (causal dense):
```python
def causal_mask(w):
    # assume w is a kernel matrix
    return jnp.triu(w)

MaskedLinear = eqx.map_variables(eqx.nn.Linear, map_in_fn=lambda m: m, where=lambda leaf: False)  # you’d refine where/map for just kernel
```
(You’d likely refine `where` to drill to just the `.weight` field, e.g. `where=lambda leaf: isinstance(leaf, jax.Array) and leaf.ndim == 2`.)

- Bit-sign weights:
```python
BitLinear = eqx.map_variables(
    eqx.nn.Linear,
    map_in_fn=lambda w: jax.tree_util.tree_map(jnp.sign, w),
    where=eqx.is_array
)
layer = BitLinear(in_size, out_size, key=key)
y = layer(x)   # uses sign-mapped weights on-the-fly
```

- Returned updated module when `mutate=True`:
```python
NormLinear = eqx.map_variables(
    eqx.nn.Linear,
    map_in_fn=lambda w: w,
    map_out_fn=lambda w: w / (jnp.linalg.norm(w, axis=0, keepdims=True) + 1e-8),
    mutate=True,
)

layer = NormLinear(in_size, out_size, key=key)
(out, layer) = layer(x)  # layer now has normalised weights
```

**Path-Aware Extension (Optional)**  
- To emulate Flax’s ability to use structural context (collection+name), you can:
  - Use `jax.tree_util.tree_map_with_path` (if available) or maintain a parallel traversal collecting paths.
  - Change `map_in_fn(leaf)` to `map_in_fn(path, leaf)`.

**Composition With Existing Transforms**  
- Order matters: mask-inside-jit vs jit-inside-mask.
  - For per-call ephemeral mapping: wrap the class first, then apply `filter_jit` to the call site or method.
  - If mapping is cheap, doing it inside the jitted function is fine; expensive mapping might be better precomputed once (then you’d mutate and capture updated module; subsequent calls skip mapping).

**Caveats**  
- No RNG scoping: if your `map_in_fn` is random you must supply deterministic seeds yourself.
- `mutate=True` returns a new module; you must capture it. Silent “in-place” updates are impossible due to frozen dataclass semantics.
- Filtering granularity: Without explicit collections you rely on predicates; document clearly for users.
- Potential recompilations: If you wrap a class then change only arrays’ values, JAX won’t recompile; but if `map_in_fn` changes shapes you may trigger recompilation.

**Possible Future Enhancements**  
- Support `methods=None` to auto-detect all non-magic callables.
- Add a functional variant `eqx.map_variables_fn` for already-instantiated modules.
- Provide built-in predicates: `eqx.trainable` vs `eqx.non_trainable` once defined, or match name substrings using `tree_pformat` metadata.

**Next Steps**  
- Decide minimal first version (likely: only ephemeral mapping, no mutate).
- Implement skeleton + tests mirroring Flax’s tied autoencoder and bitweight examples.
- Add docs section “Transforming Parameters (map_variables analogue)”.

Let me know if you’d like me to actually add the file + tests now, or refine the API before coding.
