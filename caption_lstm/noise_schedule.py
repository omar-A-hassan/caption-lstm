"""
MDLM-style masked diffusion noise schedule.

Forward process: mask tokens with probability (1 - alpha(t)) where alpha(t) = 1 - t.
  t=0 → fully clean (no masks)
  t=1 → fully masked

Training objective: cross-entropy on masked positions, reweighted by 1/t.

Inference: iterative confidence-based unmasking over num_steps steps.

Reference: Sahoo et al., "Simple and Effective Masked Diffusion Language Models" (2024).
"""

import torch
import torch.nn.functional as F


# BERT's [MASK] token — already in vocabulary at index 103, no need to add it
MASK_TOKEN_ID = 103


def alpha(t: torch.Tensor) -> torch.Tensor:
    """
    Linear noise schedule: alpha(t) = 1 - t.
    alpha(0) = 1  → fully clean
    alpha(1) = 0  → fully masked
    """
    return 1.0 - t


def sample_t(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample noise levels uniformly from (0, 1), avoiding exact endpoints."""
    return torch.rand(batch_size, device=device).clamp(1e-4, 1.0 - 1e-4)


def apply_mask(
    token_ids: torch.Tensor,
    t: torch.Tensor,
    mask_token_id: int = MASK_TOKEN_ID,
):
    """
    Forward diffusion: mask each token independently with probability (1 - alpha(t)).

    Args:
        token_ids:     (B, L) clean token ids
        t:             (B,) noise levels in [0, 1]
        mask_token_id: id of the [MASK] token

    Returns:
        noisy_ids: (B, L) — some positions replaced by [MASK]
        mask:      (B, L) bool — True where token was masked
    """
    B, L = token_ids.shape
    mask_prob = (1.0 - alpha(t)).unsqueeze(1).expand(B, L)  # (B, L)
    mask = torch.bernoulli(mask_prob).bool()
    noisy_ids = token_ids.clone()
    noisy_ids[mask] = mask_token_id
    return noisy_ids, mask


def mdlm_loss(
    logits: torch.Tensor,
    x0: torch.Tensor,
    mask: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    MDLM training loss: reweighted cross-entropy on masked positions only.

    Reweighting lambda(t) = 1/t upweights the near-clean regime (small t)
    where the prediction task is hardest and the gradient signal richest.

    Args:
        logits: (B, L, V) raw logits from denoiser
        x0:     (B, L)    clean token ids (targets)
        mask:   (B, L)    bool — True where position is masked (loss computed here)
        t:      (B,)      noise levels

    Returns:
        Scalar loss.
    """
    B, L, V = logits.shape

    if not mask.any():
        return logits.sum() * 0.0

    # Per-sample weight, broadcast to sequence dimension
    lambda_t = (1.0 / t.clamp(min=1e-4)).unsqueeze(1).expand(B, L)  # (B, L)

    # Cross-entropy without reduction
    targets = x0.clone()
    targets[~mask] = -100  # ignore unmasked positions

    loss_per_pos = F.cross_entropy(
        logits.reshape(B * L, V),
        targets.reshape(B * L),
        ignore_index=-100,
        reduction='none',
    ).reshape(B, L)

    weighted = loss_per_pos * lambda_t * mask.float()
    return weighted.sum() / mask.float().sum().clamp(min=1.0)


class InferenceSampler:
    """
    Iterative confidence-based unmasking for MDLM inference (MaskGIT-style).

    Steps (num_steps iterations from t=1 → t=0):
      1. Run denoiser on the current (partially masked) sequence.
      2. Optionally apply classifier-free guidance.
      3. Reveal the top-k most confident masked positions.
      4. Repeat until no masks remain.
    """

    def __init__(
        self,
        num_steps: int = 20,
        guidance_scale: float = 2.0,
        mask_token_id: int = MASK_TOKEN_ID,
        temperature: float = 1.0,
    ):
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.mask_token_id = mask_token_id
        self.temperature = temperature

    @torch.no_grad()
    def sample(
        self,
        model,
        image_tokens: torch.Tensor,
        max_len: int = 32,
    ) -> torch.Tensor:
        """
        Generate a token sequence by iterative unmasking.

        Args:
            model:        ViLDiffusionVLM — must expose .denoise(ids, t, image_tokens)
            image_tokens: (B, N_patches, D) pre-encoded image features
            max_len:      length of the output sequence

        Returns:
            ids: (B, max_len) generated token ids
        """
        B = image_tokens.shape[0]
        device = image_tokens.device

        ids = torch.full((B, max_len), self.mask_token_id, dtype=torch.long, device=device)
        null_image = torch.zeros_like(image_tokens)
        use_cfg = abs(self.guidance_scale - 1.0) > 1e-3

        for step in range(self.num_steps, 0, -1):
            t = torch.full((B,), step / self.num_steps, device=device)

            logits_cond = model.denoise(ids, t, image_tokens)

            if use_cfg:
                logits_uncond = model.denoise(ids, t, null_image)
                logits = logits_uncond + self.guidance_scale * (logits_cond - logits_uncond)
            else:
                logits = logits_cond

            if self.temperature != 1.0:
                logits = logits / self.temperature

            probs = torch.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1)           # (B, L)
            confidence = probs.max(dim=-1).values  # (B, L)

            is_masked = ids == self.mask_token_id  # (B, L)
            if not is_masked.any():
                break

            for b in range(B):
                masked_pos = is_masked[b].nonzero(as_tuple=True)[0]
                n_masked = masked_pos.shape[0]
                if n_masked == 0:
                    continue

                n_reveal = n_masked if step == 1 else max(1, n_masked // step)

                top_local = confidence[b][masked_pos].topk(min(n_reveal, n_masked)).indices
                ids[b, masked_pos[top_local]] = pred[b, masked_pos[top_local]]

        return ids
