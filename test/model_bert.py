import pytest
import torch
import jax
import jax.numpy as jnp
import equinox as eqx
from jx import Darray
from jx.models.bert import BertEmbeddings
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings as TorchBertEmbeddings


def test_bert_embeddings_equivalence():
    """Test that JAX BertEmbeddings produces equivalent outputs to PyTorch implementation."""
    # Setup
    seq_len = 9
    jax_key = jax.random.key(42)
    bert_config = BertConfig()
    
    # Create test inputs
    test_inputs = {
        'input_ids_jax': jnp.arange(0, seq_len),
        'position_ids_jax': jnp.arange(0, seq_len),
        'token_type_ids_jax': jnp.zeros(seq_len, dtype=int),
        'input_ids_torch': torch.arange(0, seq_len).reshape(1, -1),
        'position_ids_torch': torch.arange(0, seq_len).reshape(1, -1),
        'token_type_ids_torch': torch.zeros(seq_len, dtype=torch.long).reshape(1, -1),
    }
    
    # Create PyTorch model in eval mode
    torch_bert_model = TorchBertEmbeddings(bert_config)
    torch_bert_model.eval()
    
    # Create JAX model
    jx_model = BertEmbeddings(bert_config, rngs=jax_key)
    
    # Copy weights from PyTorch to JAX model
    jx_model = eqx.tree_at(
        lambda m: m.word_embeddings.weight,
        jx_model,
        Darray(jnp.asarray(torch_bert_model.word_embeddings.weight.cpu().detach().numpy()))
    )
    jx_model = eqx.tree_at(
        lambda m: m.position_embeddings.weight,
        jx_model,
        Darray(jnp.asarray(torch_bert_model.position_embeddings.weight.cpu().detach().numpy()))
    )
    jx_model = eqx.tree_at(
        lambda m: m.token_type_embeddings.weight,
        jx_model,
        Darray(jnp.asarray(torch_bert_model.token_type_embeddings.weight.cpu().detach().numpy()))
    )
    jx_model = eqx.tree_at(
        lambda m: m.LayerNorm.weight,
        jx_model,
        Darray(jnp.asarray(torch_bert_model.LayerNorm.weight.cpu().detach().numpy()))
    )
    jx_model = eqx.tree_at(
        lambda m: m.LayerNorm.bias,
        jx_model,
        Darray(jnp.asarray(torch_bert_model.LayerNorm.bias.cpu().detach().numpy()))
    )
    
    # Set JAX model to inference mode
    jx_model = eqx.nn.inference_mode(jx_model)
    
    # Run forward pass
    jx_output = jx_model(
        test_inputs['input_ids_jax'],
        test_inputs['position_ids_jax'],
        test_inputs['token_type_ids_jax'],
        rngs=jax.random.key(123)  # Provide RNGs for inference mode
    )
    
    torch_output = torch_bert_model(
        test_inputs['input_ids_torch'],
        token_type_ids=test_inputs['token_type_ids_torch'],
        position_ids=test_inputs['position_ids_torch']
    )
    
    # Compare outputs (squeeze PyTorch output to remove batch dimension)
    torch_output_squeezed = torch_output.squeeze(0).detach().numpy()
    
    # Assert shapes match
    assert jx_output.shape == torch_output_squeezed.shape, f"Shape mismatch: JAX {jx_output.shape} vs PyTorch {torch_output_squeezed.shape}"
    
    # Assert outputs are close (allowing for floating point differences)
    assert jnp.allclose(jx_output, torch_output_squeezed, atol=1e-4, rtol=1e-4), \
        f"Outputs differ beyond tolerance:\nJAX: {jx_output[:3]}\nPyTorch: {torch_output_squeezed[:3]}"
