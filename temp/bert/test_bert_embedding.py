import jax
import jax.numpy as jnp
import equinox as eqx
from jx import Darray
from jx.models.bert import BertEmbeddings
from transformers import BertConfig, BertTokenizer
from transformers.models.bert.modeling_bert import BertEmbeddings as TorchBertEmbeddings
from jax.tree_util import tree_structure


def main():
    key = jax.random.key(42)
    config = BertConfig()
    torch_model = TorchBertEmbeddings(config)
    jx_model = BertEmbeddings(config, rngs = key) 

    word_matrix = torch_model.word_embeddings.weight
    position_matrix = torch_model.position_embeddings.weight
    token_type_matrix = torch_model.token_type_embeddings.weight
    input_ids = jnp.arange(1, 10)
    print(f"DEBUGPRINT[37]: test_bert_embedding.py:20: input_ids={input_ids}")
    position_ids = jnp.arange(1, 10)
    print(f"DEBUGPRINT[38]: test_bert_embedding.py:22: position_ids={position_ids}")
    token_type_ids = jnp.zeros(9, dtype = int)
    print(f"DEBUGPRINT[39]: test_bert_embedding.py:24: token_type_ids={token_type_ids}")


    jx_model = eqx.tree_at(
        lambda m: m.word_embeddings.weight,
        jx_model,
        Darray(jnp.asarray(word_matrix.cpu().detach().numpy()))
    )
    jx_model = eqx.tree_at(
        lambda m: m.position_embeddings.weight,
        jx_model,
        Darray(jnp.asarray(position_matrix.cpu().detach().numpy()))
    )
    jx_model: BertEmbeddings = eqx.tree_at(
        lambda m: m.token_type_embeddings.weight,
        jx_model,
        Darray(jnp.asarray(token_type_matrix.cpu().detach().numpy()))
    )

    jx_output = jx_model(input_ids, position_ids, token_type_ids)
    print(f"DEBUGPRINT[40]: test_bert_embedding.py:44: jx_output={jx_output}")


if __name__ == "__main__":
    main()
