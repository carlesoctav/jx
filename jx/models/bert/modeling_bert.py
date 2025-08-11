from typing import Any

import equinox as eq
import jax
import jax.numpy as jnp
from equinox import field
from jaxtyping import Array, Float, Int, PRNGKeyArray
from jx import nn
from jx.nn import functional as F
from transformers.models.bert.configuration_bert import BertConfig


Pytree = Any


class BertEmbeddings(eq.Module):
    word_embeddings: nn.Embedding
    position_embeddings: nn.Embedding
    token_type_embeddings: nn.Embedding
    LayerNorm: nn.LayerNorm
    dropout: nn.Dropout

    def __init__(
        self,
        config: BertConfig,
        *,
        dtype: jnp.dtype = jnp.float16,
        rngs: PRNGKeyArray,
    ):
        word_key, position_key, token_type_key, layer_norm_key, dropout_key = (
            jax.random.split(rngs, 5)
        )

        self.word_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            dtype=dtype,
            key=word_key,
        )
        self.position_embeddings = nn.Embedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=config.hidden_size,
            dtype=dtype,
            key=position_key,
        )

        self.token_type_embeddings = nn.Embedding(
            num_embeddings=config.type_vocab_size,
            embedding_dim=config.hidden_size,
            dtype=dtype,
            key=token_type_key,
        )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)
        self.dropout = nn.Dropout(
            p = config.hidden_dropout_prob,
            inference= False,
        )
    #outer dim should be seq len instead of batch 
    def __call__(
        self,
        input_ids: Int[Array, " seq_len"],
        position_ids: Int[Array, " seq_len"],
        token_type_ids: Int[Array, " seq_len"],
        *, 
        state: Pytree | None = None,
        rngs: PRNGKeyArray | None = None,
    ) -> Float[Array, "seq_len hidden_size"]:
        inputs_embeddings = jax.vmap(self.word_embeddings)(input_ids)
        position_embeddings = jax.vmap(self.position_embeddings)(position_ids)
        token_type_embeddings = jax.vmap(self.token_type_embeddings)(token_type_ids)

        embeddings = inputs_embeddings + token_type_embeddings + position_embeddings
        print(f"DEBUGPRINT[41]: modeling_bert.py:66: embeddings={embeddings}")

        embeddings = jax.vmap(self.LayerNorm)(embeddings) 
        embeddings = self.dropout(embeddings)
        return embeddings



class BertSelfAttention(eq.Module):
    query: nn.Linear
    value: nn.Linear
    key: nn.Linear
    dropout: nn.Dropout
    num_attention_heads: int = field(static=True)
    attention_head_size: int = field(static=True)
    all_head_size: int = field(static=True)

    def __init__(
        self,
        config: BertConfig,
        *,
        dtype: jnp.dtype = jnp.float16,
        rngs: PRNGKeyArray,
    ):
        q_key, v_key, k_key = jax.random.split(rngs, 3)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size})"
                "is not a multiple of the number of attention"
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.dropout = nn.Dropout(config.hidden_dropout_prob)


        #array
        self.query = nn.Linear(config.hidden_size, self.all_head_size, rngs=q_key)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, rngs=k_key)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, rngs=v_key)

    def __call__(
        self,
        hidden_states: Float[Array, "seq_len hidden_size"],
        attention_mask: Array | None = None,
    ) -> Array:
        seq_length = hidden_states[0]
        q = jax.vmap(self.query)(hidden_states)
        k = jax.vmap(self.key)(hidden_states)
        v = jax.vmap(self.value)(hidden_states)

        q_heads = q.reshape(-1, self.num_attention_heads, self.attention_head_size)
        k_heads = k.reshape(-1, self.num_attention_heads, self.attention_head_size)
        v_heads = v.reshape(-1, self.num_attention_heads, self.attention_head_size)

        #TODO: think about attention mask later
        attn_heads = jax.vmap(F.dot_product_attention, in_axes=1, out_axes=1)(
            q_heads, k_heads, v_heads
        )

        attn = attn_heads.reshape(seq_length, -1) # seq_len, hidden_size
        return attn

class BertSelfOutput(eq.Module):
    dense: nn.Linear
    LayerNorm: nn.LayerNorm
    dropout: nn.Dropout

    def __init__(
        self,
        config: BertConfig,
        *, 
        dtype: jnp.dtype = jnp.float16,
        rngs: PRNGKeyArray,
    ):
        dense_key = jax.random.split(rngs, 1)[0]
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, rngs=dense_key)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(
        self,
        hidden_states: Float[Array, "seq_len hidden_size"],
        input_tensor: Float[Array, "seq_len hidden_size"]
    ):
        hidden_states = jax.vmap(self.dense)(hidden_states) 
        hidden_states = self.dropoout(hidden_states)
        hidden_states = jax.vmap(self.LayerNorm)(hidden_states + input_tensor)
        return hidden_states


    
class BertAttention(eq.Module):
    self: BertSelfAttention
    output: BertSelfOutput

    def __init__(
        self,
        config: BertConfig,
        *,
        dtype: jnp.dtype = jnp.float16,
        rngs: PRNGKeyArray,

    ):
        self.self = BertSelfAttention(
            config,
            dtype = dtype,
            rngs = rngs
        )

        self.output = BertSelfOutput(
            config,
            dtype = dtype,
            rngs = rngs
        )

    def __call__(
        self,
        hidden_states: Float[Array, "seq_len hidden_size"], 
    ):
        self_output = self.self(
            hidden_states
        )
        attention_output = self.output(self_output,  hidden_states)
        return attention_output




class BertIntermediate(eq.Module):
    dense: nn.Linear
    # intermediate_act_fn: Callable  todo: think about this later

    def __init__(
        self,
        config: BertConfig,
        *, 
        dtype = jnp.float16,
        rngs: PRNGKeyArray 
    ):
        dense_key = jax.random.split(rngs, 1)
        self.dense = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            rngs = dense_key
        )

    def __call__(
        self,
        hidden_states: Float[Array, "seq_len hidden_size"]
    )-> Float[Array, "seq_len intermediate_size"]:
        hidden_states = jax.vmap(self.dense)(hidden_states)
        hidden_states = jax.nn.gelu(hidden_states) 

        return hidden_states


class BertOutput(eq.Module):
    dense: nn.Linear
    LayerNorm: nn.LayerNorm
    dropout: nn.Dropout

    def __init__(
        self,
        config: BertConfig,
        *,
        dtype:jnp.dtype = jnp.float16,
        rngs: PRNGKeyArray,

    ):
        dense_key, layer_norm_key, dropout_key = jax.random.split(rngs, 3)
        self.dense = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            rngs = dense_key
        )
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size,
            eps = config.layer_norm_eps,
            rngs = layer_norm_key
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def __call__(
        self,
        hidden_states: Float[Array, "seq_len intermediate_size"],
        input_tensor: Float[Array, "seq_len hidden_size"]
    )-> Float[Array, "seq_len hidden_size"]:
        hidden_states = jax.vmap(self.dense)(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

#thinking about layer
class BertLayer(eq.Module):
    attention: BertAttention
    intermediate: BertIntermediate
    output: BertOutput


    def __init__(
        self,
        config: BertConfig, 
        *,
        dtype: jnp.dtype = jnp.float16,
        rngs: PRNGKeyArray,
    ):
        attention_key, intermediate_key, output_key = jax.random.split(rngs, 3)

        self.attention = BertAttention(config, rngs = attention_key)
        self.intermediate = BertIntermediate(config, rngs = intermediate_key)
        self.output = BertOutput(config, rngs = output_key)
    
    def __call__(
        self,
        hidden_states: Float[Array, "seq_len hidden_size"]
    ):
        attention_output = self.attention(hidden_states)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output



class BertEncoder(eq.Module):
    layer: list[BertLayer]



    def __init__(
        self,
        config: BertConfig, 
        *,
        dtype: jnp.dtype = jnp.float16,
        rngs: PRNGKeyArray
    ):
        encoder_key = jax.random.split(rngs, config.num_hidden_layers)
        for i in range(config.num_hidden_layers):
            self.layer.append(BertEncoder(config, rngs = encoder_key[i]))

    def __call__(
        self,
        hidden_states: Float[Array, "seq_len hidden_size"]
    )-> Float[Array, "seq_len hidden_size"]:
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states)
        return hidden_states



class BertModel(eq.Module):
    embeddings: BertEmbeddings
    encoder: BertEncoder
    config: BertConfig = field(static = True)


    def __init__(
        self,
        config: BertConfig, 
        *,
        dtype: jnp.dtype = jnp.float16,
        rngs: PRNGKeyArray,

    ):
        embedding_key, encoder_key = jax.random.split(rngs, 2)
        self.embeddings = BertEmbeddings(config, rngs = embedding_key)
        self.encoder = BertEmbeddings(config, rngs = encoder_key)


    def __call__(
        self,
        input_ids: Int[Array, " seq_len"],
        position_ids: Int[Array, " seq_len"],
        token_type_ids: Int[Array, " seq_len"],
    ):
        hidden_states = self.embeddings(input_ids, position_ids, token_type_ids)
        hidden_states = self.encoder(hidden_states)

        return hidden_states
