import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
import equinox as eq
from equinox import field
from jx import nn
from transformers.models.bert.configuration_bert import BertConfig


class BertEmbedding(eq.Module):
    word_embedding: nn.Embedding
    position_embedding: nn.Embedding
    token_type_embedding: nn.Embedding
    LayerNorm: nn.LayerNorm
    dropout: nn.Dropout
    config: BertConfig = field(static=True)

    def __init__(
        self,
        config: BertConfig,
        dtype: jnp.dtype = jnp.float32,  # the dtype of the computation
        param_dtype: jnp.dtype = jnp.float32,
        *,
        key: PRNGKeyArray,
    ):
        self.word_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=key,
        )
        self.position_embeddings = nn.Embedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=key,
        )

        self.token_type_embeddings = nn.Embedding(
            num_embeddings=config.type_vocab_size,
            embedding_dim=config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=key,
        )

        self.LayerNorm = nn.LayerNorm(
            num_embedding_dim=config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=key,
        )
        self.dropout = nn.Dropout(rate=config.hidden_dropout_prob, rngs=key)

    def __call__(x: Array):
        pass
