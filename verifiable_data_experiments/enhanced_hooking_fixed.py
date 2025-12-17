"""
Fixed version of enhanced_hooking that handles multi-GPU device placement.

This module provides activation steering utilities with proper device handling
for models split across multiple GPUs using device_map="auto".

Supports multiple model architectures including Gemma, Llama, Qwen, etc.
"""

import torch
from typing import Dict, Optional, Tuple, Any


# Global state for hooks
_hooks = []


def clear_hooks(model):
    """Remove all registered hooks from the model."""
    global _hooks
    for hook in _hooks:
        hook.remove()
    _hooks.clear()


def get_model_layers(model):
    """
    Get the transformer layers from a model, handling different architectures.
    
    Supports:
    - Gemma (model.model.language_model.model.layers)
    - Llama (model.model.layers)
    - Qwen (model.model.layers)
    - GPT-2 style (model.transformer.h)
    """
    # Try Gemma-specific path first (most nested)
    if hasattr(model, 'model'):
        if hasattr(model.model, 'language_model'):
            if hasattr(model.model.language_model, 'model'):
                if hasattr(model.model.language_model.model, 'layers'):
                    return model.model.language_model.model.layers
        # Try standard Llama/Qwen path
        if hasattr(model.model, 'layers'):
            return model.model.layers
    
    # Try GPT-2 style
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    
    # Try direct layers attribute
    if hasattr(model, 'layers'):
        return model.layers
    
    raise ValueError(
        f"Could not find model layers. Model type: {type(model).__name__}. "
        f"Available attributes: {[attr for attr in dir(model) if not attr.startswith('_')][:20]}"
    )


def add_activations_and_generate(
    model,
    tokens: Dict[str, torch.Tensor],
    specificpos_layer_activations: Dict[int, Dict[int, torch.Tensor]],
    continuouspos_layer_activations: Dict[int, torch.Tensor],
    sampling_kwargs: Dict[str, Any],
    add_at: str = "end",
    score_on_token: Optional[int] = None
) -> Tuple[torch.Tensor, Tuple]:
    """
    Generate text with activation steering applied at specific positions.
    
    Args:
        model: The language model
        tokens: Input tokens dict with 'input_ids' and optionally 'attention_mask'
        specificpos_layer_activations: Dict mapping layer_idx -> {position -> activation_vector}
        continuouspos_layer_activations: Dict mapping layer_idx -> activation_vector (applied continuously)
        sampling_kwargs: Generation parameters
        add_at: When to add activations ("end" for last token position)
        score_on_token: Optional specific token position to score
        
    Returns:
        Tuple of (generated_ids, scores)
    """
    global _hooks
    
    # Get model layers using architecture-aware helper
    layers = get_model_layers(model)
    
    # Register hooks for each layer that needs steering
    all_layer_indices = set(specificpos_layer_activations.keys()) | set(continuouspos_layer_activations.keys())
    
    for layer_idx in all_layer_indices:
        if layer_idx >= len(layers):
            raise ValueError(f"Layer index {layer_idx} out of range (model has {len(layers)} layers)")
        
        layer = layers[layer_idx]
        
        # Get activation vectors for this layer
        pos_activations = specificpos_layer_activations.get(layer_idx, {})
        continuous_activation = continuouspos_layer_activations.get(layer_idx, None)
        
        def make_hook(pos_acts, cont_act):
            """Create a hook function with the activation vectors in closure."""
            def hook(module, args, output):
                """
                Hook function that adds steering vectors to layer outputs.
                
                Handles multi-GPU setups by moving steering vectors to match
                the device of the hidden states.
                """
                # Output format depends on whether it's a tuple
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                # Get the device of the actual hidden states
                target_device = hidden_states.device
                
                batch_size, seq_len, hidden_dim = hidden_states.shape
                
                # Apply position-specific activations
                for position, activation_vector in pos_acts.items():
                    # Move activation vector to the correct device and ensure contiguous
                    act_vec = activation_vector.to(device=target_device, dtype=hidden_states.dtype)
                    if not act_vec.is_contiguous():
                        act_vec = act_vec.contiguous()
                    
                    # Handle negative indexing
                    if position < 0:
                        actual_pos = seq_len + position
                    else:
                        actual_pos = position
                    
                    # Only apply if position is valid
                    if 0 <= actual_pos < seq_len:
                        # Broadcast activation vector across batch if needed
                        if act_vec.dim() == 1:
                            hidden_states[:, actual_pos, :] = hidden_states[:, actual_pos, :] + act_vec
                        else:
                            hidden_states[:, actual_pos, :] = hidden_states[:, actual_pos, :] + act_vec[0]
                
                # Apply continuous activation (to all positions)
                if cont_act is not None:
                    cont_vec = cont_act.to(device=target_device, dtype=hidden_states.dtype)
                    if not cont_vec.is_contiguous():
                        cont_vec = cont_vec.contiguous()
                    
                    if cont_vec.dim() == 1:
                        # Broadcast across batch and sequence
                        hidden_states = hidden_states + cont_vec.unsqueeze(0).unsqueeze(0)
                    else:
                        hidden_states = hidden_states + cont_vec[0].unsqueeze(0).unsqueeze(0)
                
                # Return in the same format as input
                if isinstance(output, tuple):
                    return (hidden_states,) + output[1:]
                else:
                    return hidden_states
            
            return hook
        
        # Register the hook
        hook_handle = layer.register_forward_hook(make_hook(pos_activations, continuous_activation))
        _hooks.append(hook_handle)
    
    # Generate with hooks active
    with torch.no_grad():
        outputs = model.generate(**tokens, **sampling_kwargs)
    
    # Extract generated IDs and scores
    if hasattr(outputs, 'sequences'):
        generated_ids = outputs.sequences
    else:
        generated_ids = outputs
    
    scores = outputs.scores if hasattr(outputs, 'scores') else ()
    
    return generated_ids, scores
