import functools as ft
import types
from collections.abc import Callable
from typing import Any

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
        """ not need to split/partition the module because we dont compute gradient wrt to pamas to predict the outspec of module"""
        if not prev_predict_spec_fn:
            def __predict_out_spec__(self, *args, **kwargs):
                """the idea is to predict the out_spec given the spec_map_out_fn"""
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
            output = orig(mapped_self, *args, **kwargs)
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
