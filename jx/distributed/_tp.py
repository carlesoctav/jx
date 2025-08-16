from typing import Callable, Dict, Any
import equinox as eq
from equinox import field


def map_variable():
  pass


class ModelParallelismWrapper(eq.Module):
  model_axis_name: str = field(static = True)
  module_fn: Callable[..., eq.Module]
  module_kwargs: dict[str, Any]


  def __init__(
    *args,
    **kwargs
  ):
    module = map_variable
    pass
