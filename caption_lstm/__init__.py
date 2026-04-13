import sys
from pathlib import Path

# Allow imports that expect top-level `vislstm` while running from repo root.
_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if _SRC_ROOT.exists():
    _src_root = str(_SRC_ROOT)
    if _src_root not in sys.path:
        sys.path.insert(0, _src_root)

from caption_lstm.model import ViLCap, ViLCapConfig
from caption_lstm.decoder import CaptionDecoder, CaptionDecoderConfig
from caption_lstm.fusion import FiLMGenerator
from caption_lstm.tokenizer import CaptionTokenizer
from caption_lstm.mbr_decoder import MBRCaptionDecoder
from caption_lstm.clip_loss import CLIPProjectionHeads, CLIPContrastiveLoss

# Optional diffusion stack: keep classic imports working even when
# diffusion-specific dependencies are not installed.
try:
    from caption_lstm.diffusion_vlm import ViLDiffusionVLM, ViLDiffusionConfig
    from caption_lstm.denoiser import BidirMLSTMDenoiser, DenoiserConfig, SinusoidalTimestepEmbedding
    from caption_lstm.noise_schedule import (
        MASK_TOKEN_ID,
        alpha,
        sample_t,
        apply_mask,
        mdlm_loss,
        InferenceSampler,
    )
except Exception:
    ViLDiffusionVLM = None
    ViLDiffusionConfig = None
    BidirMLSTMDenoiser = None
    DenoiserConfig = None
    SinusoidalTimestepEmbedding = None
    MASK_TOKEN_ID = None
    alpha = None
    sample_t = None
    apply_mask = None
    mdlm_loss = None
    InferenceSampler = None

__all__ = [
    "ViLCap",
    "ViLCapConfig",
    "CaptionDecoder",
    "CaptionDecoderConfig",
    "FiLMGenerator",
    "CaptionTokenizer",
    "MBRCaptionDecoder",
    "CLIPProjectionHeads",
    "CLIPContrastiveLoss",
    # Optional diffusion stack
    "ViLDiffusionVLM",
    "ViLDiffusionConfig",
    "BidirMLSTMDenoiser",
    "DenoiserConfig",
    "SinusoidalTimestepEmbedding",
    "MASK_TOKEN_ID",
    "alpha",
    "sample_t",
    "apply_mask",
    "mdlm_loss",
    "InferenceSampler",
]
