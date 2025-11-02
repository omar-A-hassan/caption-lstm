"""
CLIP-style contrastive loss for image captioning.

Implements the InfoNCE contrastive loss used in CLIP (Radford et al., 2021)
to align visual and textual representations in a shared embedding space.

Reference: https://github.com/openai/CLIP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CLIPProjectionHeads(nn.Module):
    """
    Projection heads for CLIP contrastive learning.

    Projects image and text features into a shared embedding space
    where semantically similar pairs should have high cosine similarity.
    """

    def __init__(self, image_dim: int, text_dim: int, embed_dim: int = 512):
        """
        Args:
            image_dim: Dimension of visual features from encoder
            text_dim: Dimension of text features from decoder
            embed_dim: Shared embedding space dimension (default: 512)
        """
        super().__init__()
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.embed_dim = embed_dim

        # Linear projections to shared space
        self.image_proj = nn.Linear(image_dim, embed_dim, bias=False)
        self.text_proj = nn.Linear(text_dim, embed_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize projection weights."""
        nn.init.xavier_uniform_(self.image_proj.weight)
        nn.init.xavier_uniform_(self.text_proj.weight)

    def forward(self, image_features, text_features):
        """
        Project features to shared embedding space.

        Args:
            image_features: (batch_size, image_dim) - Visual features
            text_features: (batch_size, text_dim) - Text features

        Returns:
            image_embed: (batch_size, embed_dim) - L2 normalized image embeddings
            text_embed: (batch_size, embed_dim) - L2 normalized text embeddings
        """
        # Project to shared space
        image_embed = self.image_proj(image_features)  # (B, embed_dim)
        text_embed = self.text_proj(text_features)      # (B, embed_dim)

        # L2 normalize (crucial for cosine similarity)
        image_embed = F.normalize(image_embed, p=2, dim=-1)
        text_embed = F.normalize(text_embed, p=2, dim=-1)

        return image_embed, text_embed


class CLIPContrastiveLoss(nn.Module):
    """
    CLIP-style contrastive loss (InfoNCE) with in-batch negatives.

    Computes symmetric contrastive loss between image and text embeddings:
    - Image-to-text: Each image should match its caption
    - Text-to-image: Each caption should match its image

    Uses in-batch negatives: In a batch of size N, each sample has (N-1)
    negative pairs automatically.

    Formula:
        sim(i, j) = (image[i] · text[j]) / temperature
        loss_i2t[i] = -log( exp(sim(i, i)) / Σ_j exp(sim(i, j)) )
        loss_t2i[j] = -log( exp(sim(j, j)) / Σ_i exp(sim(i, j)) )
        loss = (loss_i2t + loss_t2i) / 2
    """

    def __init__(self, temperature_init: float = 0.07, learnable_temperature: bool = True):
        """
        Args:
            temperature_init: Initial temperature value (default: 0.07 from CLIP)
            learnable_temperature: If True, temperature is a learnable parameter
        """
        super().__init__()

        # Learnable temperature parameter (stored as log-scale for stability)
        # CLIP uses: logit_scale = log(1 / temperature)
        if learnable_temperature:
            self.logit_scale = nn.Parameter(
                torch.ones([]) * np.log(1.0 / temperature_init)
            )
        else:
            self.register_buffer(
                'logit_scale',
                torch.tensor(np.log(1.0 / temperature_init))
            )

    def forward(self, image_embed, text_embed):
        """
        Compute CLIP contrastive loss.

        Args:
            image_embed: (batch_size, embed_dim) - L2 normalized image embeddings
            text_embed: (batch_size, embed_dim) - L2 normalized text embeddings

        Returns:
            loss: Scalar tensor - Symmetric contrastive loss
        """
        batch_size = image_embed.shape[0]

        # Get temperature from learnable parameter
        temperature = self.logit_scale.exp()

        # Compute cosine similarity matrix (both are L2 normalized)
        # logits[i, j] = similarity between image[i] and text[j]
        logits_per_image = (image_embed @ text_embed.T) * temperature  # (B, B)
        logits_per_text = logits_per_image.T  # (B, B)

        # Labels: diagonal indices (image[i] should match text[i])
        labels = torch.arange(batch_size, device=image_embed.device)

        # Symmetric loss: image-to-text + text-to-image
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)

        loss = (loss_i2t + loss_t2i) / 2.0

        return loss

    def get_temperature(self):
        """Get current temperature value."""
        return self.logit_scale.exp().item()

    def get_similarity_matrix(self, image_embed, text_embed):
        """
        Compute similarity matrix for visualization/debugging.

        Args:
            image_embed: (batch_size, embed_dim)
            text_embed: (batch_size, embed_dim)

        Returns:
            similarity: (batch_size, batch_size) - Similarity scores
        """
        temperature = self.logit_scale.exp()
        similarity = (image_embed @ text_embed.T) * temperature
        return similarity
