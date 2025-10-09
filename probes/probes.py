
"""
probes.py
----------
Flexible probe definitions for classifying jailbreak success (binary label)
from model latent states.

Input expected shape: (batch, num_prompt_tokens, dim_of_residual_stream).
If you already pooled over tokens, you can pass shape (batch, dim) and
the probes will handle it transparently.

Probes:
  - LinearProbe: single linear classifier.
  - MLPProbe: multi-layer perceptron with configurable depth.
  - TransformerProbe: small Transformer encoder over the token axis with a CLS token,
    followed by a classification head.

All probes return logits (not probabilities). Use BCEWithLogitsLoss for training.

Example:
    probe = MLPProbe(input_dim=4096, hidden_dims=[1024, 256], dropout=0.1)
    logits = probe(x)  # x: (B, T, D) or (B, D)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------- Helpers ----------------------
def _ensure_2d(x: torch.Tensor, pooling: str = "mean") -> torch.Tensor:
    """
    Converts (B, T, D) -> (B, D) by pooling over tokens, or passes through (B, D).
    pooling: "mean" | "last" | "max"
    """
    if x.dim() == 3:
        B, T, D = x.shape
        if pooling == "mean":
            return x.mean(dim=1)
        elif pooling == "last":
            return x[:, -1, :]
        elif pooling == "max":
            return x.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
    elif x.dim() == 2:
        return x
    else:
        raise ValueError(f"Expected tensor of shape (B, T, D) or (B, D); got {tuple(x.shape)}")


class LayerNormFP32(nn.LayerNorm):
    """LayerNorm that runs in fp32 for numerical stability."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        orig_dtype = x.dtype
        return super().forward(x.to(torch.float32)).to(orig_dtype)


# ---------------------- Linear Probe ----------------------
class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, pooling: str = "mean", bias: bool = True):
        super().__init__()
        self.pooling = pooling
        self.fc = nn.Linear(input_dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_2d(x, pooling=self.pooling)
        return self.fc(x).squeeze(-1)  # (B,)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------- Nonlinear (MLP) Probe ----------------------
class MLPProbe(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (512,),
        activation: str = "gelu",
        dropout: float = 0.0,
        pooling: str = "mean",
        layernorm: bool = False,
    ):
        super().__init__()
        self.pooling = pooling

        act_layer = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
        }.get(activation.lower())
        if act_layer is None:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if layernorm:
                layers.append(LayerNormFP32(h))
            layers.append(act_layer())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_2d(x, pooling=self.pooling)
        return self.net(x).squeeze(-1)  # (B,)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------- Transformer Probe ----------------------
class TransformerProbe(nn.Module):
    """
    A compact Transformer encoder over the token axis.

    Accepts x of shape (B, T, D). If x is (B, D), it is treated as T=1.

    Args:
        input_dim: feature dimension D.
        n_layers: number of TransformerEncoder layers.
        n_heads:  number of attention heads.
        mlp_ratio: FFN hidden size = mlp_ratio * input_dim.
        dropout: dropout applied to attention and FFN.
        add_cls_token: whether to learn a [CLS] token for classification.
        pooling: if no CLS token, how to pool over tokens ("mean" | "last" | "max").
    """
    def __init__(
        self,
        input_dim: int,
        n_layers: int = 2,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        add_cls_token: bool = True,
        pooling: str = "mean",
    ):
        super().__init__()
        self.add_cls = add_cls_token
        self.pooling = pooling

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_heads,
            dim_feedforward=int(mlp_ratio * input_dim),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, input_dim)) if add_cls_token else None
        self.ln = LayerNormFP32(input_dim)
        self.head = nn.Linear(input_dim, 1)

        # Init cls token
        if self.cls is not None:
            nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, D) or (B, D)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, D)
        B, T, D = x.shape

        if self.cls is not None:
            cls_tokens = self.cls.expand(B, -1, -1)  # (B, 1, D)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, 1+T, D)
            # Build attention mask if provided: expect (B, T) 1 for keep, 0 for pad
            if attn_mask is not None:
                # prepend keep for CLS
                attn_mask = torch.cat([torch.ones(B, 1, device=x.device, dtype=attn_mask.dtype), attn_mask], dim=1)
                # PyTorch expects True for positions to mask out; invert
                src_key_padding_mask = ~attn_mask.bool()
            else:
                src_key_padding_mask = None
            h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
            cls_out = self.ln(h[:, 0, :])
            logits = self.head(cls_out).squeeze(-1)
            return logits  # (B,)
        else:
            if attn_mask is not None:
                src_key_padding_mask = ~attn_mask.bool()
            else:
                src_key_padding_mask = None
            h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # (B, T, D)
            pooled = _ensure_2d(h, pooling=self.pooling)
            pooled = self.ln(pooled)
            return self.head(pooled).squeeze(-1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = [
    "LinearProbe",
    "MLPProbe",
    "TransformerProbe",
]
