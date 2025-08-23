from dataclasses import dataclass, field
from typing import TypeVar

import equinox as eq
import jax
import jax.tree_util as jtu


A = TypeVar('A')


@jtu.register_dataclass
@dataclass(slots = True) #think about this later
class Darray:
    value: jax.Array  | None
    pspec: str | tuple[str, ...] | None = field(metadata=dict(static=True), default = None)



def first_from(*args: A | None, error_msg: str) -> A:
  """Return the first non-None argument.

  If all arguments are None, raise a ValueError with the given error message.

  Args:
    *args: the arguments to check
    error_msg: the error message to raise if all arguments are None
  Returns:
    The first non-None argument.
  """
  for arg in args:
    if arg is not None:
      return arg
  raise ValueError(error_msg)


