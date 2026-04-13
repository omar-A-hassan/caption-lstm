"""
ViLDiffusionVLM — Vision-Language Model with masked diffusion text generation.

Architecture (joint-sequence design):
  1. VisionLSTM encoder  : image → (B, N_patches, encoder_dim)
  2. Image projection    : → (B, N_patches, model_dim)
  3. Text embedding      : noisy token ids → (B, L, model_dim)
  4. Timestep embedding  : t → (B, model_dim), added to text positions only
  5. Modality embeddings : distinguish image tokens from text tokens
  6. Joint sequence      : [image tokens | text tokens] → (B, N+L, model_dim)
  7. BidirMLSTMDenoiser  : → (B, N+L, model_dim)
  8. Output projection   : text positions → (B, L, vocab_size)

Training (MDLM):
  - Sample t ~ U(0,1), mask each text token with prob (1-alpha(t))
  - Model predicts clean tokens at masked positions
  - Loss: reweighted CE (weight = 1/t) on masked positions
  - Classifier-free guidance: drop image tokens with p_uncond=0.1 during training

Inference:
  - Start fully masked, iteratively unmask most-confident positions (MaskGIT-style)
  - Apply CFG at every denoising step
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer

from vision_lstm.vision_lstm import VisionLSTM
from caption_lstm.denoiser import BidirMLSTMDenoiser, DenoiserConfig, SinusoidalTimestepEmbedding
from caption_lstm.noise_schedule import (
    MASK_TOKEN_ID,
    sample_t,
    apply_mask,
    mdlm_loss,
    InferenceSampler,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ViLDiffusionConfig:
    # --- Vision encoder ---
    encoder_dim: int = 192
    encoder_depth: int = 24
    encoder_input_shape: tuple = (3, 224, 224)
    encoder_patch_size: int = 16
    encoder_drop_path_rate: float = 0.0
    encoder_pretrained_path: str = None
    freeze_encoder: bool = True         # freeze for first N steps (handled in training loop)

    # --- Model ---
    model_dim: int = 512
    num_denoiser_blocks: int = 6
    num_heads: int = 8
    proj_factor: float = 2.0
    dropout: float = 0.1

    # --- Text ---
    # BERT-base vocab (30522). [MASK]=103 is already in it — no extra tokens needed.
    vocab_size: int = 30522
    max_text_length: int = 32
    tokenizer_model: str = "bert-base-uncased"

    # --- Diffusion / inference ---
    p_uncond: float = 0.1               # CFG dropout probability during training
    num_inference_steps: int = 20
    guidance_scale: float = 2.0
    inference_temperature: float = 1.0

    @property
    def num_image_patches(self) -> int:
        h, w = self.encoder_input_shape[1], self.encoder_input_shape[2]
        return (h // self.encoder_patch_size) * (w // self.encoder_patch_size)

    @property
    def context_length(self) -> int:
        return self.num_image_patches + self.max_text_length


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ViLDiffusionVLM(nn.Module):
    """
    Vision-Language diffusion model built on bidirectional mLSTM.

    Exposes two main interfaces:
      forward()  — training step, returns loss dict
      generate() — inference, returns decoded caption strings
    """

    def __init__(self, config: ViLDiffusionConfig):
        super().__init__()
        self.config = config
        N = config.num_image_patches   # 196 for 224×224 / patch 16
        L = config.max_text_length
        D = config.model_dim

        # ------------------------------------------------------------------
        # 1. Vision encoder (pretrained VisionLSTM, optionally frozen)
        # ------------------------------------------------------------------
        self.encoder = VisionLSTM(
            dim=config.encoder_dim,
            input_shape=config.encoder_input_shape,
            patch_size=config.encoder_patch_size,
            depth=config.encoder_depth,
            output_shape=None,
            mode=None,
            pooling=None,
            drop_path_rate=config.encoder_drop_path_rate,
        )

        if config.encoder_pretrained_path is not None:
            self._load_encoder(config.encoder_pretrained_path)

        if config.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        # ------------------------------------------------------------------
        # 2. Image projection + modality embeddings
        # ------------------------------------------------------------------
        self.img_proj = nn.Linear(config.encoder_dim, D)
        # Learnable type embeddings distinguish image from text tokens
        self.img_type_embed = nn.Parameter(torch.zeros(1, 1, D))
        self.txt_type_embed = nn.Parameter(torch.zeros(1, 1, D))
        # Positional embeddings for text tokens (image patches keep ViL's own pos embed)
        self.txt_pos_embed = nn.Parameter(torch.randn(1, L, D) * 0.02)

        # ------------------------------------------------------------------
        # 3. Text embedding
        # ------------------------------------------------------------------
        self.token_embedding = nn.Embedding(config.vocab_size, D)

        # ------------------------------------------------------------------
        # 4. Timestep embedding
        # ------------------------------------------------------------------
        self.time_embed = SinusoidalTimestepEmbedding(model_dim=D)

        # ------------------------------------------------------------------
        # 5. Bidirectional mLSTM denoiser
        # ------------------------------------------------------------------
        denoiser_cfg = DenoiserConfig(
            model_dim=D,
            num_blocks=config.num_denoiser_blocks,
            num_heads=config.num_heads,
            proj_factor=config.proj_factor,
            dropout=config.dropout,
            context_length=config.context_length,
        )
        self.denoiser = BidirMLSTMDenoiser(denoiser_cfg)

        # ------------------------------------------------------------------
        # 6. Output head
        # ------------------------------------------------------------------
        self.output_norm = nn.LayerNorm(D)
        self.output_proj = nn.Linear(D, config.vocab_size, bias=False)

        # ------------------------------------------------------------------
        # 7. Tokenizer
        # ------------------------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_model)

        # ------------------------------------------------------------------
        # Initialise weights
        # ------------------------------------------------------------------
        self._init_weights()
        self._init_bert_embeddings()

        # ------------------------------------------------------------------
        # Inference sampler
        # ------------------------------------------------------------------
        self.sampler = InferenceSampler(
            num_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            mask_token_id=MASK_TOKEN_ID,
            temperature=config.inference_temperature,
        )

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.img_proj.weight)
        nn.init.zeros_(self.img_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.img_type_embed)
        nn.init.zeros_(self.txt_type_embed)

    def _init_bert_embeddings(self):
        """
        Copy BERT-base token embeddings into this model's embedding table.
        Projects 768 → model_dim if dimensions differ.
        """
        try:
            print("Initialising token embeddings from BERT...")
            bert = BertModel.from_pretrained(self.config.tokenizer_model)
            bert_emb = bert.embeddings.word_embeddings.weight.data  # (30522, 768)

            D = self.config.model_dim
            V = self.config.vocab_size
            bert_V, bert_D = bert_emb.shape

            n_copy = min(V, bert_V)

            if bert_D == D:
                self.token_embedding.weight.data[:n_copy].copy_(bert_emb[:n_copy])
            else:
                proj = nn.Linear(bert_D, D, bias=False)
                nn.init.xavier_uniform_(proj.weight)
                with torch.no_grad():
                    projected = proj(bert_emb[:n_copy])
                self.token_embedding.weight.data[:n_copy].copy_(projected)

            del bert
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print(f"  Copied {n_copy} embeddings (bert_dim={bert_D} → model_dim={D})")
        except Exception as e:
            print(f"  Warning: could not load BERT embeddings ({e}). Using random init.")

    def _load_encoder(self, path: str):
        state = torch.load(path, map_location='cpu')
        if 'model' in state:
            state = state['model']
        elif 'state_dict' in state:
            state = state['state_dict']

        # Strip any 'encoder.' prefix if saved from old ViLCap checkpoint
        state = {k.replace('encoder.', ''): v for k, v in state.items()
                 if not any(x in k for x in ['head', 'decoder', 'fusion', 'output'])}
        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        print(f"Loaded encoder from {path}  (missing={len(missing)}, unexpected={len(unexpected)})")

    # ------------------------------------------------------------------
    # Core forward pass (used in both training and inference denoising)
    # ------------------------------------------------------------------

    def denoise(
        self,
        token_ids: torch.Tensor,
        t: torch.Tensor,
        image_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single denoising pass over the joint [image | text] sequence.

        Args:
            token_ids:    (B, L) — current (noisy/masked) text token ids
            t:            (B,)  — noise levels in [0, 1]
            image_tokens: (B, N, model_dim) — projected image patch features

        Returns:
            logits: (B, L, vocab_size)
        """
        B, L = token_ids.shape
        N = image_tokens.shape[1]

        # Image side: add modality embedding
        img = image_tokens + self.img_type_embed          # (B, N, D)

        # Text side: embed tokens, add positional + modality + timestep
        txt = self.token_embedding(token_ids)             # (B, L, D)
        txt = txt + self.txt_pos_embed[:, :L, :]
        txt = txt + self.txt_type_embed
        t_emb = self.time_embed(t).unsqueeze(1)           # (B, 1, D)
        txt = txt + t_emb                                 # broadcast over L

        # Joint sequence: image prefix then text
        joint = torch.cat([img, txt], dim=1)              # (B, N+L, D)

        # Denoising
        hidden = self.denoiser(joint)                     # (B, N+L, D)

        # Extract text positions and project to vocab
        text_hidden = hidden[:, N:, :]                    # (B, L, D)
        text_hidden = self.output_norm(text_hidden)
        return self.output_proj(text_hidden)              # (B, L, vocab_size)

    # ------------------------------------------------------------------
    # Image encoding (shared between train and generate)
    # ------------------------------------------------------------------

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to projected patch tokens.

        Args:
            images: (B, 3, H, W)

        Returns:
            (B, N_patches, model_dim)
        """
        with torch.no_grad() if not self.encoder.training else torch.enable_grad():
            patch_feats = self.encoder(images)            # (B, N, encoder_dim)
        return self.img_proj(patch_feats)                 # (B, N, model_dim)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(self, images: torch.Tensor, captions) -> dict:
        """
        Training step.

        Args:
            images:   (B, 3, H, W)
            captions: either
                - list[str] of length B
                - torch.LongTensor of shape (B, L) with token IDs

        Returns:
            dict with 'loss' (scalar) and diagnostic keys.
        """
        device = images.device
        B = images.shape[0]

        # Tokenise captions unless already provided as token IDs.
        if torch.is_tensor(captions):
            x0 = captions.to(device, dtype=torch.long)
        else:
            encoded = self.tokenizer(
                captions,
                padding='max_length',
                truncation=True,
                max_length=self.config.max_text_length,
                return_tensors='pt',
                add_special_tokens=True,
            )
            x0 = encoded['input_ids'].to(device)         # (B, L) clean token ids

        # Keep batch-size bookkeeping aligned even when caption tensors are passed.
        B = x0.shape[0]

        # Encode image
        image_tokens = self.encode_image(images)          # (B, N, D)

        # Classifier-free guidance: randomly null-out image for p_uncond fraction
        if self.training and self.config.p_uncond > 0:
            drop = torch.rand(B, device=device) < self.config.p_uncond
            image_tokens = image_tokens.clone()
            image_tokens[drop] = 0.0

        # Sample noise level and apply masking
        t = sample_t(B, device)                           # (B,)
        noisy_ids, mask = apply_mask(x0, t)               # (B, L), (B, L)

        # Denoising forward pass
        logits = self.denoise(noisy_ids, t, image_tokens) # (B, L, V)

        # MDLM loss
        loss = mdlm_loss(logits, x0, mask, t)

        # Diagnostics
        with torch.no_grad():
            n_masked = mask.float().sum()
            n_total = mask.numel()
            pred_correct = (logits.argmax(-1) == x0)[mask].float().mean() if mask.any() else torch.tensor(0.0)

        return {
            'loss': loss,
            'mask_fraction': (n_masked / n_total).item(),
            'masked_accuracy': pred_correct.item(),
            'mean_t': t.mean().item(),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        num_steps: int = None,
        guidance_scale: float = None,
        temperature: float = None,
    ) -> list[str]:
        """
        Generate captions for a batch of images.

        Args:
            images:         (B, 3, H, W)
            num_steps:      overrides config.num_inference_steps
            guidance_scale: overrides config.guidance_scale
            temperature:    overrides config.inference_temperature

        Returns:
            List of B caption strings.
        """
        self.eval()

        # Allow per-call overrides without mutating config
        sampler = self.sampler
        if any(x is not None for x in [num_steps, guidance_scale, temperature]):
            sampler = InferenceSampler(
                num_steps=num_steps or self.config.num_inference_steps,
                guidance_scale=guidance_scale if guidance_scale is not None else self.config.guidance_scale,
                mask_token_id=MASK_TOKEN_ID,
                temperature=temperature or self.config.inference_temperature,
            )

        image_tokens = self.encode_image(images)          # (B, N, D)
        token_ids = sampler.sample(
            model=self,
            image_tokens=image_tokens,
            max_len=self.config.max_text_length,
        )                                                  # (B, L)

        captions = self.tokenizer.batch_decode(
            token_ids.cpu().tolist(),
            skip_special_tokens=True,
        )
        return captions

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        print("Encoder frozen.")

    def unfreeze_encoder(self, lr_scale: float = 0.1):
        """Unfreeze the encoder. Caller should use a lower lr for encoder params."""
        for p in self.encoder.parameters():
            p.requires_grad_(True)
        print(f"Encoder unfrozen (use lr_scale≈{lr_scale} for encoder parameter group).")

    def encoder_parameters(self):
        return self.encoder.parameters()

    def denoiser_parameters(self):
        """All parameters except the encoder."""
        encoder_ids = {id(p) for p in self.encoder.parameters()}
        return [p for p in self.parameters() if id(p) not in encoder_ids]

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters() if not trainable_only else filter(lambda p: p.requires_grad, self.parameters())
        return sum(p.numel() for p in params)
