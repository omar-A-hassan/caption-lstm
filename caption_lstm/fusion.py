import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMGenerator(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) Generator with Attention Pooling.

    Generates gamma (scale) and beta (shift) parameters from visual features
    to condition the decoder via feature-wise affine transformations.

    FIXED: Instead of destructive mean pooling, we use learnable attention
    pooling to preserve spatial information while producing a global context.

    Based on: "FiLM: Visual Reasoning with a General Conditioning Layer"
    (Perez et al., 2018)
    """

    def __init__(self, visual_dim: int, decoder_dim: int):
        """
        Args:
            visual_dim: Dimension of visual features from encoder
            decoder_dim: Dimension of decoder embeddings
        """
        super().__init__()

        # === FIX: Learnable Attention Pooling ===
        # Instead of mean(dim=1), we learn which tokens are important.
        self.attn_query = nn.Parameter(torch.randn(1, 1, visual_dim) * 0.02)
        self.attn_proj = nn.Linear(visual_dim, 1)  # Produces attention scores

        # MLP to generate FiLM parameters
        self.fc1 = nn.Linear(visual_dim, decoder_dim)
        self.fc2 = nn.Linear(decoder_dim, 2 * decoder_dim)  # gamma + beta

        # Initialize fc2 to output gamma≈1, beta≈0 (identity initialization)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def _attention_pool(self, x):
        """
        Learnable attention pooling over tokens.
        
        Args:
            x: (B, N, D) - Token-level features
        
        Returns:
            pooled: (B, D) - Global feature vector
        """
        # Compute attention scores: (B, N, 1)
        scores = self.attn_proj(x)  # Simple linear projection
        attn_weights = F.softmax(scores, dim=1)  # (B, N, 1)
        
        # Weighted sum of tokens
        pooled = (x * attn_weights).sum(dim=1)  # (B, D)
        return pooled

    def forward(self, visual_features):
        """
        Generate FiLM parameters from visual features.

        Args:
            visual_features: (batch_size, N, visual_dim) or (batch_size, visual_dim)

        Returns:
            gamma: (batch_size, decoder_dim) - scale parameters
            beta: (batch_size, decoder_dim) - shift parameters
        """
        # Pool if we have token-level features
        if visual_features.dim() == 3:
            # === FIX: Use attention pooling instead of mean ===
            visual_features = self._attention_pool(visual_features)  # (B, N, D) → (B, D)

        # Generate FiLM parameters through MLP
        x = F.relu(self.fc1(visual_features))  # (B, decoder_dim)
        film_params = self.fc2(x)  # (B, 2*decoder_dim)

        # Split into gamma and beta
        gamma, beta = torch.chunk(film_params, 2, dim=-1)

        return gamma, beta

