import torch
import torch.nn as nn


class SimpleFusion(nn.Module):
    """
    Simple fusion module for injecting visual context into decoder.

    Following the Bi-LSTM paper's approach: uses an Add/Merge layer
    to inject visual features into the first token of the decoder.
    No cross-attention needed.
    """

    def __init__(self, visual_dim: int, decoder_dim: int):
        """
        Args:
            visual_dim: Dimension of visual features from encoder
            decoder_dim: Dimension of decoder embeddings
        """
        super().__init__()

        # Linear projection to match decoder dimension
        self.visual_proj = nn.Linear(visual_dim, decoder_dim)

    def forward(self, visual_features):
        """
        Project visual features to decoder dimension.

        Args:
            visual_features: (batch_size, visual_dim)

        Returns:
            context: (batch_size, decoder_dim)
        """
        # Project visual features to decoder space
        context = self.visual_proj(visual_features)

        return context
