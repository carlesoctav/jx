import pytest
import torch
import jax
import jax.numpy as jnp
import equinox as eqx
from jx import Darray
from jx.models.bert import BertEmbeddings, BertAttention, BertEncoder
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings as TorchBertEmbeddings, BertAttention as TorchBertAttention, BertLayer as TorchBertLayer


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
    jx_model = BertEmbeddings(bert_config, key=jax_key)
    
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
        key=jax.random.key(123)  # Provide key for inference mode
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


def test_bert_attention_equivalence():
    """Test that JAX BertAttention produces equivalent outputs to PyTorch implementation."""
    seq_len = 8
    jax_key = jax.random.key(42)
    bert_config = BertConfig()
    hidden_size = bert_config.hidden_size
    
    hidden_states_jax = jax.random.normal(jax.random.key(123), (seq_len, hidden_size))
    hidden_states_torch = torch.tensor(jnp.asarray(hidden_states_jax), dtype=torch.float32).unsqueeze(0)  # Add batch dim
    
    bert_config._attn_implementation = "eager" 
    torch_attention = TorchBertAttention(bert_config)
    torch_attention.eval()
    
    jx_attention = BertAttention(bert_config, key=jax_key)
    
    jx_attention = eqx.tree_at(
        lambda m: m.self.query.weight,
        jx_attention,
        Darray(jnp.asarray(torch_attention.self.query.weight.cpu().detach().numpy()))  # No transpose needed
    )
    jx_attention = eqx.tree_at(
        lambda m: m.self.query.bias,
        jx_attention,
        Darray(jnp.asarray(torch_attention.self.query.bias.cpu().detach().numpy()))
    )
    jx_attention = eqx.tree_at(
        lambda m: m.self.key.weight,
        jx_attention,
        Darray(jnp.asarray(torch_attention.self.key.weight.cpu().detach().numpy()))
    )
    jx_attention = eqx.tree_at(
        lambda m: m.self.key.bias,
        jx_attention,
        Darray(jnp.asarray(torch_attention.self.key.bias.cpu().detach().numpy()))
    )
    jx_attention = eqx.tree_at(
        lambda m: m.self.value.weight,
        jx_attention,
        Darray(jnp.asarray(torch_attention.self.value.weight.cpu().detach().numpy()))
    )
    jx_attention = eqx.tree_at(
        lambda m: m.self.value.bias,
        jx_attention,
        Darray(jnp.asarray(torch_attention.self.value.bias.cpu().detach().numpy()))
    )
    
    # Output layer weights
    jx_attention = eqx.tree_at(
        lambda m: m.output.dense.weight,
        jx_attention,
        Darray(jnp.asarray(torch_attention.output.dense.weight.cpu().detach().numpy()))
    )
    jx_attention = eqx.tree_at(
        lambda m: m.output.dense.bias,
        jx_attention,
        Darray(jnp.asarray(torch_attention.output.dense.bias.cpu().detach().numpy()))
    )
    jx_attention = eqx.tree_at(
        lambda m: m.output.LayerNorm.weight,
        jx_attention,
        Darray(jnp.asarray(torch_attention.output.LayerNorm.weight.cpu().detach().numpy()))
    )
    jx_attention = eqx.tree_at(
        lambda m: m.output.LayerNorm.bias,
        jx_attention,
        Darray(jnp.asarray(torch_attention.output.LayerNorm.bias.cpu().detach().numpy()))
    )
    
    # Set JAX model to inference mode
    jx_attention = eqx.nn.inference_mode(jx_attention)
    
    # Run forward pass
    try:
        jx_output = jx_attention(
            hidden_states_jax,
            key=jax.random.key(456),
            inference=True
        )
        
        print(f"DEBUGPRINT[48]: test_model_bert.py:165: jx_output={jx_output}")
        torch_output = torch_attention(hidden_states_torch)[0] 
        print(f"DEBUGPRINT[47]: test_model_bert.py:171: torch_output={torch_output}")
        
        torch_output_squeezed = torch_output.squeeze(0).detach().numpy()
        
        assert jx_output.shape == torch_output_squeezed.shape, f"Shape mismatch: JAX {jx_output.shape} vs PyTorch {torch_output_squeezed.shape}"
        
        assert jnp.allclose(jx_output, torch_output_squeezed, atol=1e-3, rtol=1e-3), \
            f"Outputs differ beyond tolerance:\nJAX: {jx_output[:2]}\nPyTorch: {torch_output_squeezed[:2]}"
            
    except Exception as e:
        pytest.fail(f"BertAttention test failed with error: {e}")


def test_bert_encoder_single_layer_equivalence():
    """Test that JAX BertEncoder with one layer produces equivalent outputs to PyTorch implementation."""
    seq_len = 6
    jax_key = jax.random.key(42)
    
    # Create config with only 1 layer for simpler testing
    bert_config = BertConfig(
        vocab_size=1000,
        hidden_size=32,
        num_hidden_layers=1,  # Single layer
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=512,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0
    )
    
    # Create test input
    hidden_size = bert_config.hidden_size
    hidden_states_jax = jax.random.normal(jax.random.key(123), (seq_len, hidden_size))
    hidden_states_torch = torch.tensor(jnp.asarray(hidden_states_jax), dtype=torch.float32).unsqueeze(0)  # Add batch dim
    
    # Create models
    bert_config._attn_implementation = "eager"
    torch_layer = TorchBertLayer(bert_config)
    torch_layer.eval()
    
    jx_encoder = BertEncoder(bert_config, key=jax_key)
    
    # Copy weights from PyTorch layer to JAX encoder's first (and only) layer
    # Since we have a single layer, copy directly from torch_layer to jx_encoder.layer[0]
    jx_encoder = eqx.tree_at(
        lambda m: m.layer[0].attention.self.query.weight,
        jx_encoder,
        Darray(jnp.asarray(torch_layer.attention.self.query.weight.cpu().detach().numpy()))
    )
    jx_encoder = eqx.tree_at(
        lambda m: m.layer[0].attention.self.query.bias,
        jx_encoder,
        Darray(jnp.asarray(torch_layer.attention.self.query.bias.cpu().detach().numpy()))
    )
    jx_encoder = eqx.tree_at(
        lambda m: m.layer[0].attention.self.key.weight,
        jx_encoder,
        Darray(jnp.asarray(torch_layer.attention.self.key.weight.cpu().detach().numpy()))
    )
    jx_encoder = eqx.tree_at(
        lambda m: m.layer[0].attention.self.key.bias,
        jx_encoder,
        Darray(jnp.asarray(torch_layer.attention.self.key.bias.cpu().detach().numpy()))
    )
    jx_encoder = eqx.tree_at(
        lambda m: m.layer[0].attention.self.value.weight,
        jx_encoder,
        Darray(jnp.asarray(torch_layer.attention.self.value.weight.cpu().detach().numpy()))
    )
    jx_encoder = eqx.tree_at(
        lambda m: m.layer[0].attention.self.value.bias,
        jx_encoder,
        Darray(jnp.asarray(torch_layer.attention.self.value.bias.cpu().detach().numpy()))
    )
    
    # Attention output weights
    jx_encoder = eqx.tree_at(
        lambda m: m.layer[0].attention.output.dense.weight,
        jx_encoder,
        Darray(jnp.asarray(torch_layer.attention.output.dense.weight.cpu().detach().numpy()))
    )
    jx_encoder = eqx.tree_at(
        lambda m: m.layer[0].attention.output.dense.bias,
        jx_encoder,
        Darray(jnp.asarray(torch_layer.attention.output.dense.bias.cpu().detach().numpy()))
    )
    jx_encoder = eqx.tree_at(
        lambda m: m.layer[0].attention.output.LayerNorm.weight,
        jx_encoder,
        Darray(jnp.asarray(torch_layer.attention.output.LayerNorm.weight.cpu().detach().numpy()))
    )
    jx_encoder = eqx.tree_at(
        lambda m: m.layer[0].attention.output.LayerNorm.bias,
        jx_encoder,
        Darray(jnp.asarray(torch_layer.attention.output.LayerNorm.bias.cpu().detach().numpy()))
    )
    
    # Intermediate weights
    jx_encoder = eqx.tree_at(
        lambda m: m.layer[0].intermediate.dense.weight,
        jx_encoder,
        Darray(jnp.asarray(torch_layer.intermediate.dense.weight.cpu().detach().numpy()))
    )
    jx_encoder = eqx.tree_at(
        lambda m: m.layer[0].intermediate.dense.bias,
        jx_encoder,
        Darray(jnp.asarray(torch_layer.intermediate.dense.bias.cpu().detach().numpy()))
    )
    
    # Output weights
    jx_encoder = eqx.tree_at(
        lambda m: m.layer[0].output.dense.weight,
        jx_encoder,
        Darray(jnp.asarray(torch_layer.output.dense.weight.cpu().detach().numpy()))
    )
    jx_encoder = eqx.tree_at(
        lambda m: m.layer[0].output.dense.bias,
        jx_encoder,
        Darray(jnp.asarray(torch_layer.output.dense.bias.cpu().detach().numpy()))
    )
    jx_encoder = eqx.tree_at(
        lambda m: m.layer[0].output.LayerNorm.weight,
        jx_encoder,
        Darray(jnp.asarray(torch_layer.output.LayerNorm.weight.cpu().detach().numpy()))
    )
    jx_encoder = eqx.tree_at(
        lambda m: m.layer[0].output.LayerNorm.bias,
        jx_encoder,
        Darray(jnp.asarray(torch_layer.output.LayerNorm.bias.cpu().detach().numpy()))
    )
    
    # Set to inference mode
    jx_encoder = eqx.nn.inference_mode(jx_encoder)
    
    # Test with simpler call first
    print("Testing BertEncoder call...")
    try:
        # Try without key first
        jx_output = jx_encoder(hidden_states_jax, inference=True)
        print("Call without key succeeded!")
    except Exception as e:
        print(f"Call without key failed: {e}")
        # Try with key
        try:
            jx_output = jx_encoder(
                hidden_states_jax,
                key=jax.random.key(456),
                inference=True
            )
            print("Call with key succeeded!")
        except Exception as e2:
            print(f"Call with key also failed: {e2}")
            raise e2
        
        with torch.no_grad():
            torch_output = torch_layer(hidden_states_torch)[0].squeeze(0)  # Remove batch dim
        
        torch_output_array = torch_output.detach().numpy()
        
        # Assert shapes match
        assert jx_output.shape == torch_output_array.shape, f"Shape mismatch: JAX {jx_output.shape} vs PyTorch {torch_output_array.shape}"
        
        # Assert outputs are close
        assert jnp.allclose(jx_output, torch_output_array, atol=1e-3, rtol=1e-3), \
            f"Outputs differ beyond tolerance:\nJAX: {jx_output[:2]}\nPyTorch: {torch_output_array[:2]}"
            
        print("BertEncoder single layer test passed!")
            
    except Exception as e:
        pytest.fail(f"BertEncoder test failed with error: {e}")
