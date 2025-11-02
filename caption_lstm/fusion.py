import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMGenerator(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) Generator.

    Generates gamma (scale) and beta (shift) parameters from visual features
    to condition the decoder via feature-wise affine transformations.

    Based on: "FiLM: Visual Reasoning with a General Conditioning Layer"
    (Perez et al., 2018)

    FiLM applies: output = gamma ⊙ x + beta
    where gamma and beta are learned functions of the conditioning input (visual features).
    """

    def __init__(self, visual_dim: int, decoder_dim: int):
        """
        Args:
            visual_dim: Dimension of visual features from encoder
            decoder_dim: Dimension of decoder embeddings
        """
        super().__init__()

        # MLP to generate FiLM parameters
        self.fc1 = nn.Linear(visual_dim, decoder_dim)
        self.fc2 = nn.Linear(decoder_dim, 2 * decoder_dim)  # gamma + beta

        # Initialize fc2 to output gamma≈1, beta≈0 (identity initialization)
        # This ensures FiLM starts as an identity transform and learns from there
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

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
            visual_features = visual_features.mean(dim=1)  # (B, N, D) → (B, D)

        # Generate FiLM parameters through MLP
        x = F.relu(self.fc1(visual_features))  # (B, decoder_dim)
        film_params = self.fc2(x)  # (B, 2*decoder_dim)

        # Split into gamma and beta
        gamma, beta = torch.chunk(film_params, 2, dim=-1)

        return gamma, beta
