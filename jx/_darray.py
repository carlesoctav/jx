import jax

import typing as tp
from dataclasses import dataclass, field
import jax.tree_util as jtu
from jax import P


@jtu.register_dataclass
@dataclass(slots = True)
class Darray:
    value: jax.Array
    pspec: tp.Optional[P] = field(metadata=dict(static=True), default = None)
