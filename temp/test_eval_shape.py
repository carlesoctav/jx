import jax
from jx import nn 
import functools as ft
import equinox as eq
import jax.numpy as jnp


def eval_shape(fn, *args, **kwargs):
    def _fn(_array, _static):
        _fun, _args, _kwargs = eq.combine(_array, _static)
        _out = _fun(*_args, **_kwargs)
        _array, _static = eq.partition(_out, eq.is_array)
        return _array, _static 

    array, static = eq.partition((fn, args, kwargs), eq.is_array)
    passable_fn = ft.partial(_fn, _static = static)
    shape_array, shape_static = jax.eval_shape(passable_fn, array)
    return eq.combine(shape_array, shape_static)




key = jax.random.key(10)
linear = nn.Linear(10, 10, key = key, weight_pspec = (None, "model"), dtype = jnp.float16)
shape = eval_shape(linear, jax.random.normal(key, (10,))) 
shape = eq.filter_eval_shape(linear, jax.random.normal(key, (10,))) 


shape = nn.param_shapes(linear)
print(f"DEBUGPRINT[107]: test_eval_shape.py:28: shape={shape}")
