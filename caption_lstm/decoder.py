import torch
import torch.nn as nn
from dataclasses import dataclass

from src.vislstm.modules.xlstm.blocks.mlstm.layer import mLSTMLayer, mLSTMLayerConfig


@dataclass
class CaptionDecoderConfig:
    vocab_size: int
    embedding_dim: int = 512
    num_blocks: int = 3
    num_heads: int = 4
    proj_factor: float = 2.0
    conv1d_kernel_size: int = 4
    qkv_proj_blocksize: int = 4
    dropout: float = 0.2
    max_length: int = 50


class CaptionDecoder(nn.Module):
    """
    Causal mLSTM decoder for image captioning.

    Uses unidirectional mLSTM blocks for autoregressive caption generation.
    Following the Bi-LSTM paper architecture with 3 blocks and dropout pattern.
    """

    def __init__(self, config: CaptionDecoderConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)

        # Positional embedding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, config.max_length, config.embedding_dim) * 0.02
        )

        # mLSTM blocks (causal/unidirectional)
        self.blocks = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        # Dropout pattern: 0.2, 0.2, 0.5 (from Bi-LSTM paper)
        dropout_rates = [0.2, 0.2, 0.5]

        for i in range(config.num_blocks):
            # Create mLSTM layer config (unidirectional, causal)
            mlstm_config = mLSTMLayerConfig(
                embedding_dim=config.embedding_dim,
                conv1d_kernel_size=config.conv1d_kernel_size,
                qkv_proj_blocksize=config.qkv_proj_blocksize,
                num_heads=config.num_heads,
                proj_factor=config.proj_factor,
                bidirectional=False,  # Causal decoder
                quaddirectional=False,
                context_length=config.max_length,
                dropout=0.0,  # Internal dropout
                _num_blocks=config.num_blocks,
            )
            mlstm_config.__post_init__()

            self.blocks.append(mLSTMLayer(mlstm_config))
            self.dropout_layers.append(nn.Dropout(dropout_rates[min(i, len(dropout_rates) - 1)]))

        # Output projection to vocabulary
        self.output_proj = nn.Linear(config.embedding_dim, config.vocab_size)

        self.reset_parameters()

    def forward(self, token_ids, film_gamma=None, film_beta=None):
        """
        Forward pass for decoder with FiLM conditioning.

        Args:
            token_ids: Token IDs (batch_size, seq_len)
            film_gamma: FiLM scale parameters (batch_size, embedding_dim)
                If provided, applies feature-wise scaling before each block
            film_beta: FiLM shift parameters (batch_size, embedding_dim)
                If provided, applies feature-wise shifting before each block

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape

        # Token embedding
        x = self.token_embedding(token_ids)  # (B, S, D)

        # Add positional embedding
        x = x + self.pos_embedding[:, :seq_len, :]

        # Pass through mLSTM blocks with FiLM conditioning
        for i, (block, dropout) in enumerate(zip(self.blocks, self.dropout_layers)):
            # Apply FiLM conditioning BEFORE block (as in FiLM paper)
            if film_gamma is not None and film_beta is not None:
                # gamma, beta: (B, D) → expand to (B, 1, D) → broadcast to (B, S, D)
                gamma = film_gamma.unsqueeze(1)  # (B, 1, D)
                beta = film_beta.unsqueeze(1)    # (B, 1, D)
                x = gamma * x + beta  # Feature-wise linear modulation

            # Residual connection
            x = x + dropout(block(x, block_idx=i))

        # Project to vocabulary
        logits = self.output_proj(x)

        return logits

    def generate(self, bos_token_id, eos_token_id, pad_token_id,
                 film_gamma=None, film_beta=None, max_length=50, temperature=1.0, top_k=None):
        """
        Autoregressive generation with FiLM conditioning.

        Args:
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID
            film_gamma: FiLM scale parameters (batch_size, embedding_dim)
            film_beta: FiLM shift parameters (batch_size, embedding_dim)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter

        Returns:
            generated_ids: (batch_size, max_length)
        """
        batch_size = film_gamma.shape[0] if film_gamma is not None else 1
        device = film_gamma.device if film_gamma is not None else next(self.parameters()).device

        # Start with BOS token
        generated = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length - 1):
            # Forward pass with FiLM conditioning
            logits = self.forward(generated, film_gamma=film_gamma, film_beta=film_beta)

            # Get logits for next token (last position)
            next_token_logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Update finished sequences
            finished = finished | (next_token.squeeze(-1) == eos_token_id)

            # Replace finished sequences with pad token
            next_token[finished] = pad_token_id

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if all sequences are finished
            if finished.all():
                break

        return generated

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)
