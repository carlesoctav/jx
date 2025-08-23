import functools as ft
import types
from collections.abc import Callable
from typing import Any, get_args, get_origin

import dataclasses as dc
import equinox as eq

class HasPredictMethodOutSpecMixin:
    def __predict_out_spec__(self, *args, **kwargs):
        pass
  

Pytree = Any

def map_variables(
    cls: type[eq.Module] | eq.Module,
    map_in_fn, 
    predict_spec_fn = None,
    name: str = "",
    *,
    where = eq.is_array,
    methods = ("__call__",)
)-> type[eq.Module]:
    if not isinstance(cls, type) or not issubclass(cls, eq.Module):
        raise TypeError("First argument must be a subclass of Module")

    def _wrap_predict_out_spec(prev_predict_spec_fn: Callable | None):
        if not prev_predict_spec_fn:
            def __predict_out_spec__(self, *args, **kwargs):
                mapped = predict_spec_fn(self, *args, **kwargs)
                return mapped 

            return __predict_out_spec__
        else:
            @ft.wraps(prev_predict_spec_fn)
            def wrapped(self, *args, **kwargs):
                mapped = prev_predict_spec_fn(self, *args, **kwargs)
                mapped = predict_spec_fn(mapped, *args, **kwargs)
                return mapped 

            return wrapped


    def _wrap_method(name, orig_fn):
        @ft.wraps(orig_fn)
        def wrapped(self, *args, **kwargs):
            to_map, others = eq.partition(self, where)
            mapped = map_in_fn(to_map)
            mapped_self = eq.combine(mapped, others)
            output = orig_fn(mapped_self, *args, **kwargs)
            return output
        return wrapped

    dct = {}
    for m in methods:
        if not hasattr(cls, m):
            raise ValueError(f"{cls.__name__} doesn't have {m} method")
        orig = getattr(cls, m)

        if isinstance(orig, types.FunctionType | types.MethodType):
            func = orig.__func__ if isinstance(orig, types.MethodType) else orig
        else:
            raise TypeError(f"{m} is not types.MethodType or types.FunctionType")

        dct[m] = _wrap_method(m, func)

    if  predict_spec_fn:
        predict_out_spec_method = None
        if hasattr(cls, "__predict_out_spec__"):
            predict_out_spec_method = getattr(cls, "__predict_out_spec__")

        dct["__predict_out_spec__"] = _wrap_predict_out_spec(predict_out_spec_method)


    subclasses = (cls, HasPredictMethodOutSpecMixin) if predict_spec_fn else (cls)
    New = type(f"{name}{cls.__name__}", subclasses, dct)
    New.__doc__ = (cls.__doc__ or "") + f"\n\n[Wrapped for {name} by map_variables]"
    return New


def _is_module_type(t: Any) -> bool:
    return isinstance(t, type) and issubclass(t, eq.Module)


def _module_field_types(cls: type[eq.Module]):
    out = []
    if not dc.is_dataclass(cls):
        return out
    for f in dc.fields(cls):
        t = f.type
        origin = get_origin(t)
        if origin is None:
            if _is_module_type(t):
                out.append((f.name, t))
        else:
            args = get_args(t)
            for a in args:
                if _is_module_type(a):
                    out.append((f.name, a))
    return out


def has_module_children(cls: type[eq.Module]) -> bool:
    return len(_module_field_types(cls)) > 0


def iter_module_type(
    module_type: type[eq.Module],
    *,
    include_root: bool = False,
):
    """Yield (path, type) for all eq.Module-typed fields.

    - If include_root=True, also yields ((), module_type).
    - Path elements are field names (no container indices).
    """

    def _maybe_map(t_owner: type[eq.Module], field_name: str, t_child: type[eq.Module]) -> type[eq.Module]:
        mapping = getattr(t_owner, "__type_transform_map__", None)
        if isinstance(mapping, dict):
            return mapping.get((field_name, t_child), t_child)
        return t_child

    if include_root:
        yield ((), module_type)

    def _visit(t: type[eq.Module], path: tuple[str, ...]):
        for name, sub_t in _module_field_types(t):
            mapped_sub_t = _maybe_map(t, name, sub_t)
            yield (path + (name,), mapped_sub_t)
            yield from _visit(mapped_sub_t, path + (name,))

    yield from _visit(module_type, ())


def transform_module_type_leaves(
    root: type[eq.Module],
    leaf_transform: Callable[[tuple[str, ...], type[eq.Module]], type[eq.Module]],
    *,
    name_prefix: str = "Transformed",
) -> type[eq.Module]:

    cache: dict[tuple[tuple[str, ...], type[eq.Module]], type[eq.Module]] = {}

    def _build_at(path: tuple[str, ...], t: type[eq.Module]) -> type[eq.Module]:
        key = (path, t)
        if key in cache:
            return cache[key]

        # Transform this type (node) itself first
        t_trans = leaf_transform(path, t)
        if t_trans is not t and not issubclass(t_trans, t):
            pass

        child_type_map: dict[tuple[str, type[eq.Module]], type[eq.Module]] = {}
        if dc.is_dataclass(t_trans):
            new_annotations = {}
            for field in dc.fields(t_trans):
                field_type = field.type
                origin = get_origin(field_type)
                if origin is None:
                    if _is_module_type(field_type):
                        new_child_t = _build_at(path + (field.name,), field_type)
                        child_type_map[(field.name, field_type)] = new_child_t
                        new_annotations[field.name] = new_child_t
                    else:
                        new_annotations[field.name] = field_type
                else:
                    args = get_args(field_type)
                    new_args = []
                    changed = False
                    for arg in args:
                        if _is_module_type(arg):
                            new_arg = _build_at(path + (field.name,), arg)
                            child_type_map[(field.name, arg)] = new_arg
                            new_args.append(new_arg)
                            changed = True
                        else:
                            new_args.append(arg)
                    if changed:
                        try:
                            new_annotations[field.name] = origin[new_args[0]] if len(new_args) == 1 else origin[tuple(new_args)]
                        except TypeError:
                            new_annotations[field.name] = field_type
                    else:
                        new_annotations[field.name] = field_type
        else:
            new_annotations = {}

        def __init__(self, *args, **kwargs):
            import sys
            class _Patch:
                def __init__(self, mapping: dict[type[eq.Module], type[eq.Module]]):
                    self.mapping = mapping
                    self._saved: list[tuple[object, str, object]] = []
                def __enter__(self):
                    mods = list(sys.modules.values())
                    for m in mods:
                        d = getattr(m, "__dict__", None)
                        if not isinstance(d, dict):
                            continue
                        for attr, val in list(d.items()):
                            if isinstance(val, type):
                                try:
                                    if issubclass(val, eq.Module) and val in self.mapping:
                                        new = self.mapping[val]
                                        self._saved.append((m, attr, val))
                                        try:
                                            setattr(m, attr, new)
                                        except Exception:
                                            pass
                                except Exception:
                                    # Non-class or weird types
                                    pass
                    return self
                def __exit__(self, exc_type, exc, tb):
                    for mod, name, orig in reversed(self._saved):
                        try:
                            setattr(mod, name, orig)
                        except Exception:
                            pass
            # Build a flat type mapping for patching from the child map and cache
            flat_map: dict[type[eq.Module], type[eq.Module]] = {}
            for (_, orig), new in child_type_map.items():
                flat_map[orig] = new
            for (p, orig), new in cache.items():
                flat_map[orig] = new
            with _Patch(flat_map):
                t.__init__(self, *args, **kwargs)

        new_name = f"{name_prefix}{t_trans.__name__}"
        new_cls = type(new_name, (t_trans,), {
            "__init__": __init__,
            "__annotations__": new_annotations,
            "__type_transform_map__": child_type_map,
        })
        cache[key] = new_cls
        return new_cls

    # Ensure we also add a mapping for the root itself
    return _build_at((), root)


def iter_module(
    module: eq.Module,
    *,
    include_root: bool = False,
):
    """Yield (path, instance) for all eq.Module instances, not just leaves.

    - Path includes indices/keys as strings for containers.
    - If include_root=True, yields ((), module) first.
    """
    if include_root:
        yield ((), module)

    def _visit(inst: eq.Module, path: tuple[str, ...]):
        for fname, _t in _module_field_types(type(inst)):
            v = getattr(inst, fname, None)

            def _yield_and_recurse(sub_inst: eq.Module, sub_path: tuple[str, ...]):
                yield (sub_path, sub_inst)
                yield from _visit(sub_inst, sub_path)

            if isinstance(v, eq.Module):
                yield from _yield_and_recurse(v, path + (fname,))
            elif isinstance(v, list):
                for i, x in enumerate(v):
                    if isinstance(x, eq.Module):
                        yield from _yield_and_recurse(x, path + (fname, str(i)))
            elif isinstance(v, tuple):
                for i, x in enumerate(v):
                    if isinstance(x, eq.Module):
                        yield from _yield_and_recurse(x, path + (fname, str(i)))
            elif isinstance(v, dict):
                for k, x in v.items():
                    if isinstance(x, eq.Module):
                        yield from _yield_and_recurse(x, path + (fname, str(k)))

    yield from _visit(module, ())

