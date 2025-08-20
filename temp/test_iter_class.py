import dataclasses
import jax
from jax import tree_util as jtu
import equinox as eq
from equinox import nn

class SubModule(eq.Module):
    linear1: nn.Linear
    linear2: nn.Linear
    def __init__(self, din, dout, key):
        self.linear1 = nn.Linear(din, dout, key=key)
        self.linear2 = nn.Linear(din, dout, key=key)

class Block(eq.Module):
    linear:nn.Linear
    submodule:SubModule
    dropout:nn.Dropout
    batch_norm:nn.BatchNorm

    def __init__(self, din, dout, *, key):
        self.linear = nn.Linear(din, dout, key=key)
        self.submodule = SubModule(din, dout, key=key)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm = nn.LayerNorm(10)

key = jax.random.key(0)
shape = eq.filter_eval_shape(Block, 2, 5, key = key)
data = dataclasses.fields(Block)
print(f"DEBUGPRINT[373]: test_iter_module.py:28: data={data}")

def iter_module(shape):
    def printing(path, leaf):
        print(f"DEBUGPRINT[372]: test_iter_module.py:30: path={path}")
        print(f"DEBUGPRINT[371]: test_iter_module.py:31: leaf={leaf}")
        if isinstance(leaf, eq.Module):
            print(f"DEBUGPRINT[368]: test_iter_module.py:29: path={path}")
            print(f"DEBUGPRINT[369]: test_iter_module.py:29: leaf={leaf}")
    jtu.tree_map_with_path(printing, shape, is_leaf = lambda x: isinstance(x, jax.Array))


# def iter_class(block):
#     fields = dataclasses.fields(block):
#     for field in fields:
#         if isinstance(field, eq.Module):
#             yield field.name, field.value 
#
#
#
# transformed = {}
# new = type(block)
# for x, y in iter_class(block):
#     x: type
#     new_x = nn.map_variables(newx, a;skdfjaslkdfs;ljdf)
#     transforrmed {
#      } => 
#
    


