import jax

import typing as tp
from dataclasses import dataclass, field
import jax.tree_util as jtu
from jax import P

A = tp.TypeVar('A')


@jtu.register_dataclass
@dataclass(slots = True)
class Darray:
    value: jax.Array
    pspec: P | None = field(metadata=dict(static=True), default = None)


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

