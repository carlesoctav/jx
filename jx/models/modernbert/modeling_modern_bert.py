import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
import equinox as eq
from equinox import field
from jx import nn
from transformers.models.bert.configuration_bert import BertConfig

class ModernBertEmbeddings(eq.Module):
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
        rngs: PRNGKeyArray,
    ):
        word_key, position_key, token_type_key, layer_norm_key, dropout_key = nn.split_rngs(rngs, 5) 
        self.word_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=word_key,
        )
        self.position_embeddings = nn.Embedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=position_key,
        )

        self.token_type_embeddings = nn.Embedding(
            num_embeddings=config.type_vocab_size,
            embedding_dim=config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=token_type_key,
        )

        self.LayerNorm = nn.LayerNorm(
        )

        self.dropout = nn.Dropout(
            rate=config.hidden_dropout_prob,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=dropout_key
        )


    def __call__(input_ids: Array, token_type_ids: Array, position_ids: Array, training: bool = False) -> Array:
        input_shape = input_ids.shape
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = jnp.arange(seq_length, dtype=jnp.int32)[None, :]

        if token_type_ids is None:
            token_type_ids = jnp.zeros(input_shape, dtype=jnp.int32)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings, training=training)

        return embeddings

