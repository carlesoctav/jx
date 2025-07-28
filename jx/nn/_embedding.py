import equinox as eq

import typing as tp
import jax.numpy as jnp
import jax
from jax import P
from jaxtyping import ArrayLike, DTypeLike, PRNGKeyArray, Int
from jx import Darray
from equinox import field, is_array_like
from jax.nn.initializers import normal

Array = jax.Array

default_init = normal()


# TODO:
# add this later
# padding_idx (int, optional) – If specified, the entries at padding_idx do not contribute to the gradient; therefore, the embedding vector at padding_idx is not updated during training, i.e. it remains as a fixed “pad”. For a newly constructed Embedding, the embedding vector at padding_idx will default to all zeros, but can be updated to another value to be used as the padding vector. max_norm (float, optional) – If given, each embedding vector with norm larger than max_norm is renormalized to have norm max_norm.
# norm_type (float, optional) – The p of the p-norm to compute for the max_norm option. Default 2.
# scale_grad_by_freq (bool, optional) – If given, this will scale gradients by the inverse of frequency of the words in the mini-batch. Default False.
# sparse (bool, optional) – If True, gradient w.r.t. weight matrix will be a sparse tensor. See Notes for more details regarding sparse gradients.
# lookup table
# [num_embedding, embedding_dim]
#TODO: dtype is useless for now
class Embedding(eq.Module, strict=True):
    weight: Darray
    num_embeddings: int = field(static=True)  # vocab size
    embedding_dim: int = field(static=True)
    dtype: DTypeLike = field(static = True)
    param_dtype: DTypeLike = field(static = True)


    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        weight_pspec: tp.Optional[P] = None,
        dtype: tp.Optional[DTypeLike] = None,
        param_dtype: DTypeLike = jnp.float32,
        key: PRNGKeyArray,
    ):

        wkey = jax.random.split(key, 1)
        wvalue = default_init(wkey, (num_embeddings, embedding_dim), param_dtype)

        self.weight = Darray(wvalue, weight_pspec)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.dtype = dtype or self.weight.value.dtype
        self.param_dtype = param_dtype 

    def __call__(self, x: Int[ArrayLike]) -> Array:
        if is_array_like(x) and jnp.shape(x) == ():
            return self.weight[x]
        else:
            raise ValueError(
                "`Embedding()(x)` should be called with a scalar index `x`. "
            )
