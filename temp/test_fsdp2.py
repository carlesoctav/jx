import equinox as eq
from jx.distributed._fsdp import fully_shard
from jx.distributed._utils import simulate_CPU_devices
from jx import nn
import jax
from jax import P
from jax import shard_map, make_mesh


simulate_CPU_devices()
key = jax.random.key(10)

FSDPLinear = fully_shard(nn.Linear, model_axis_name = "data", min_weight_size = 2)
linear = FSDPLinear(80, 10, key = key)
devices = jax.devices()
mesh = make_mesh((8,), ("data", ),  devices = devices)



def init_module(model):
    pspec = model.__predict_out_spec__()
    return pspec

if isinstance(linear, nn.Linear) and isinstance(linear, nn.HasPredictMethodOutSpecMixin):
    predict_output_spec = jax.jit(shard_map(init_module, in_specs=P(), out_specs = P(), check_vma = False, mesh = mesh))
    out = predict_output_spec(linear)

    # non_jitpredict_output_spec = shard_map(init_module, in_specs=P(), out_specs = P(), check_vma = False, mesh = mesh)
    # out = non_jitpredict_output_spec(linear)
    print(f"DEBUGPRINT[116]: test_fsdp2.py:26: out={out}")


