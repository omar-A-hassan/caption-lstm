"""
Bidirectional mLSTM denoiser for masked diffusion language modelling.

Key differences from the autoregressive CaptionDecoder:
  - bidirectional=True  : forward + backward mLSTM passes summed at every layer
  - SymmetricConv1d     : replaces CausalConv1d so the local conv kernel sees
                          both left and right context (denoiser has no causal constraint)
  - SinusoidalTimestep  : noise-level embedding added to each token before the stack
  - Pre-norm residual   : LayerNorm → mLSTMLayer → residual (more stable than post-norm)

The denoiser receives the *joint* sequence [image tokens | text tokens] produced
by ViLDiffusionVLM and returns contextualised hidden states of the same shape.
Splitting out the text positions and projecting to vocabulary is done in the parent model.
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

# Ensure the local src/ package is importable when running from repo root
_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if _SRC_ROOT.exists():
    _src_root = str(_SRC_ROOT)
    if _src_root not in sys.path:
        sys.path.insert(0, _src_root)

from vislstm.modules.xlstm.blocks.mlstm.layer import mLSTMLayer, mLSTMLayerConfig


# ---------------------------------------------------------------------------
# Non-causal conv — drop-in replacement for CausalConv1d
# ---------------------------------------------------------------------------

class SymmetricConv1d(nn.Module):
    """
    Depthwise Conv1d with symmetric (same) padding.

    CausalConv1d pads only on the left so each position only sees left context.
    For a denoiser, we want symmetric context — pad equally on both sides.
    Uses padding='same' which PyTorch computes automatically for any kernel size,
    including even sizes (kernel_size=4 is the default in mLSTMLayerConfig).

    Input/output shape: (B, S, D)  — sequence-major, matching mLSTMLayer convention.
    """

    def __init__(self, feature_dim: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(
            feature_dim,
            feature_dim,
            kernel_size,
            padding='same',
            groups=feature_dim,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D) → (B, D, S) → conv → (B, D, S) → (B, S, D)
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


def _patch_causal_conv(layer: mLSTMLayer) -> None:
    """
    Replace CausalConv1d with SymmetricConv1d in-place on an mLSTMLayer.

    Called once after construction so we don't have to fork the library.
    Both the forward and reverse conv paths are patched.
    """
    kernel_size = layer.config.conv1d_kernel_size
    inner_dim = layer.config._inner_embedding_dim

    layer.conv1d = SymmetricConv1d(inner_dim, kernel_size)

    if layer.conv1d_rev is not None:
        layer.conv1d_rev = SymmetricConv1d(inner_dim, kernel_size)


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalTimestepEmbedding(nn.Module):
    """
    Maps scalar noise level t ∈ [0, 1] → dense embedding vector.

    Uses the same sinusoidal frequencies as the original DDPM / DiT papers,
    followed by a two-layer MLP to project to model_dim.
    """

    def __init__(self, model_dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) noise levels in [0, 1]
        Returns:
            (B, model_dim) timestep embeddings
        """
        device = t.device
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )                                            # (half,)
        args = t[:, None] * freqs[None, :]           # (B, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, freq_dim)
        return self.mlp(emb)                         # (B, model_dim)


# ---------------------------------------------------------------------------
# Single denoiser block (pre-norm residual)
# ---------------------------------------------------------------------------

class DenoiserBlock(nn.Module):
    """
    Pre-LayerNorm bidirectional mLSTM block with symmetric conv.

    Structure:
        x → LayerNorm → bidir-mLSTM → Dropout → + x
    """

    def __init__(self, mlstm_config: mLSTMLayerConfig, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(mlstm_config.embedding_dim)
        self.mlstm = mLSTMLayer(mlstm_config)
        _patch_causal_conv(self.mlstm)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mlstm(self.norm(x)))


# ---------------------------------------------------------------------------
# Full denoiser stack
# ---------------------------------------------------------------------------

@dataclass
class DenoiserConfig:
    model_dim: int = 512
    num_blocks: int = 6
    num_heads: int = 8
    proj_factor: float = 2.0
    conv1d_kernel_size: int = 4
    qkv_proj_blocksize: int = 4
    dropout: float = 0.1
    # Set to image_patches + max_text_length for the joint sequence
    context_length: int = 228   # 196 patches + 32 text tokens


class BidirMLSTMDenoiser(nn.Module):
    """
    Stack of bidirectional mLSTM blocks that processes the joint
    [image tokens | noisy text tokens] sequence and returns hidden states.

    The caller (ViLDiffusionVLM) is responsible for:
      - Prepending the image prefix
      - Adding the timestep embedding to text positions
      - Slicing out the text-position outputs
      - Projecting to vocabulary logits
    """

    def __init__(self, config: DenoiserConfig):
        super().__init__()
        self.config = config

        self.blocks = nn.ModuleList()
        for i in range(config.num_blocks):
            mlstm_cfg = mLSTMLayerConfig(
                embedding_dim=config.model_dim,
                num_heads=config.num_heads,
                proj_factor=config.proj_factor,
                conv1d_kernel_size=config.conv1d_kernel_size,
                qkv_proj_blocksize=config.qkv_proj_blocksize,
                bidirectional=True,   # forward + backward mLSTM passes summed
                quaddirectional=False,
                context_length=config.context_length,
                dropout=0.0,          # dropout applied in DenoiserBlock wrapper
                _num_blocks=config.num_blocks,
            )
            mlstm_cfg.__post_init__()
            self.blocks.append(DenoiserBlock(mlstm_cfg, dropout=config.dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N_img + L, model_dim) joint image+text hidden states
        Returns:
            (B, N_img + L, model_dim) contextualised hidden states
        """
        for block in self.blocks:
            x = block(x)
        return x
