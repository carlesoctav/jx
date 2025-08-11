import torch
import jax
import jax.numpy as jnp
import equinox as eqx
from jx import Darray
from jx.models.bert import BertEmbeddings
from transformers import BertConfig, BertTokenizer
from transformers.models.bert.modeling_bert import BertEmbeddings as TorchBertEmbeddings
from jax.tree_util import tree_structure
from equinox.nn import LayerNorm 


def main():
    key = jax.random.key(42)
    config = BertConfig()
    torch_model = TorchBertEmbeddings(config)
    torch_model.eval()  # Disable dropout
    jx_model = BertEmbeddings(config, key = key) 

    word_matrix = torch_model.word_embeddings.weight
    position_matrix = torch_model.position_embeddings.weight
    token_type_matrix = torch_model.token_type_embeddings.weight
    input_ids_jax = jnp.arange(0, 9)  # Use 0-8 to be safe
    position_ids_jax = jnp.arange(0, 9)
    token_type_ids_jax = jnp.zeros(9, dtype = int)

    input_ids_torch = torch.arange(0, 9).reshape(1, -1)  # Use 0-8 to be safe
    position_ids_torch = torch.arange(0, 9).reshape(1, -1)
    token_type_ids_torch = torch.zeros(9, dtype = torch.long).reshape(1, -1)

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
    
    # Copy LayerNorm parameters as well
    jx_model = eqx.tree_at(
        lambda m: m.LayerNorm.weight,
        jx_model,
        Darray(jnp.asarray(torch_model.LayerNorm.weight.cpu().detach().numpy()))
    )
    jx_model = eqx.tree_at(
        lambda m: m.LayerNorm.bias,
        jx_model,
        Darray(jnp.asarray(torch_model.LayerNorm.bias.cpu().detach().numpy()))
    )

    jx_model = eqx.nn.inference_mode(jx_model)

    jx_output = jx_model(
        input_ids_jax,
        position_ids_jax,
        token_type_ids_jax
    )

    print(f"DEBUGPRINT[44]: test_bert_embedding.py:47: jx_output={jx_output}")
    print(f"torch_model.token_type_embeddings.weight shape: {torch_model.token_type_embeddings.weight.shape}")
    print(f"token_type_matrix shape: {token_type_matrix.shape}")
    
    # Test torch model BEFORE modifying jx_model - only pass input_ids
    try:
        torch_output_test = torch_model(input_ids_torch)
        print("PyTorch model works fine with just input_ids")
    except Exception as e:
        print(f"PyTorch model fails even with just input_ids: {e}")
        return
        
    # Test with all parameters but correct types
    try:
        torch_output_test = torch_model(
            input_ids_torch,
            token_type_ids=token_type_ids_torch,
            position_ids=position_ids_torch
        )
        print("PyTorch model works with named parameters")
    except Exception as e:
        print(f"PyTorch model fails with named parameters: {e}")
        return

    torch_output = torch_model(
        input_ids_torch,
        token_type_ids=token_type_ids_torch,
        position_ids=position_ids_torch
    )

    print(f"DEBUGPRINT[45]: test_bert_embedding.py:54: torch_output={torch_output}")

if __name__ == "__main__":
    main()
