import torch
import torch.nn as nn
from dataclasses import dataclass

from vision_lstm.vision_lstm import VisionLSTM
from caption_lstm.tokenizer import CaptionTokenizer
from caption_lstm.decoder import CaptionDecoder, CaptionDecoderConfig
from caption_lstm.fusion import SimpleFusion


@dataclass
class ViLCapConfig:
    """Configuration for ViL-Cap model."""

    # Encoder (VisionLSTM) config
    encoder_dim: int = 192
    encoder_depth: int = 24
    encoder_input_shape: tuple = (3, 224, 224)
    encoder_patch_size: int = 16
    encoder_pooling: str = "bilateral_avg"
    encoder_drop_path_rate: float = 0.0
    encoder_pretrained_path: str = None

    # Decoder config
    vocab_size: int = 30522  # BERT vocab size
    decoder_dim: int = 512
    decoder_num_blocks: int = 3
    decoder_num_heads: int = 4
    decoder_dropout: float = 0.2
    max_caption_length: int = 50

    # Tokenizer
    tokenizer_model: str = "bert-base-uncased"


class ViLCap(nn.Module):
    """
    Vision-LSTM for Image Captioning (ViL-Cap).

    Minimal modification of ViL architecture for end-to-end image captioning.
    Uses:
    - ViL encoder (bidirectional mLSTM) for visual features
    - Causal mLSTM decoder for caption generation
    - Simple fusion (no cross-attention) following Bi-LSTM paper
    """

    def __init__(self, config: ViLCapConfig):
        super().__init__()
        self.config = config

        # Vision encoder (ViL)
        self.encoder = VisionLSTM(
            dim=config.encoder_dim,
            input_shape=config.encoder_input_shape,
            patch_size=config.encoder_patch_size,
            depth=config.encoder_depth,
            output_shape=None,  # No classification head
            mode=None,  # Feature extraction mode
            pooling=None,  # Must be None in feature extraction mode
            drop_path_rate=config.encoder_drop_path_rate,
        )

        # Store pooling config for manual pooling after encoder
        self.encoder_pooling = config.encoder_pooling

        # Load pretrained weights if provided
        if config.encoder_pretrained_path is not None:
            self.load_encoder_weights(config.encoder_pretrained_path)

        # Determine visual dimension based on pooling
        if self.encoder_pooling == "bilateral_avg" or self.encoder_pooling == "bilateral_concat":
            visual_dim = config.encoder_dim if self.encoder_pooling == "bilateral_avg" else config.encoder_dim * 2
        else:
            # No pooling - will be (B, N, D) where N = num_patches
            visual_dim = config.encoder_dim

        # Fusion module (projects visual features to decoder space)
        self.fusion = SimpleFusion(
            visual_dim=visual_dim,
            decoder_dim=config.decoder_dim
        )

        # Caption decoder
        decoder_config = CaptionDecoderConfig(
            vocab_size=config.vocab_size,
            embedding_dim=config.decoder_dim,
            num_blocks=config.decoder_num_blocks,
            num_heads=config.decoder_num_heads,
            dropout=config.decoder_dropout,
            max_length=config.max_caption_length,
        )
        self.decoder = CaptionDecoder(decoder_config)

        # Tokenizer
        self.tokenizer = CaptionTokenizer(
            model_name=config.tokenizer_model,
            max_length=config.max_caption_length
        )

        # Update decoder vocab size if tokenizer has different size
        if self.tokenizer.vocab_size != config.vocab_size:
            self.decoder.token_embedding = nn.Embedding(
                self.tokenizer.vocab_size,
                config.decoder_dim
            )
            self.decoder.output_proj = nn.Linear(
                config.decoder_dim,
                self.tokenizer.vocab_size
            )
            self.decoder.config.vocab_size = self.tokenizer.vocab_size

    def load_encoder_weights(self, path):
        """Load pretrained encoder weights."""
        state_dict = torch.load(path, map_location='cpu')

        # Handle different checkpoint formats
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # Filter encoder-only weights
        encoder_state_dict = {
            k.replace('encoder.', ''): v
            for k, v in state_dict.items()
            if k.startswith('encoder.') or not any(x in k for x in ['head', 'decoder', 'fusion'])
        }

        self.encoder.load_state_dict(encoder_state_dict, strict=False)
        print(f"Loaded encoder weights from {path}")

    def encode_image(self, images):
        """
        Encode images to visual features.

        Args:
            images: (batch_size, 3, H, W)

        Returns:
            visual_features: (batch_size, encoder_dim) or (batch_size, N, encoder_dim)
        """
        # Encoder returns (B, N, D) where N = num_patches
        features = self.encoder(images)

        # Apply pooling if specified
        if self.encoder_pooling == "bilateral_avg":
            # Average of first and last token
            features = (features[:, 0] + features[:, -1]) / 2  # (B, D)
        elif self.encoder_pooling == "bilateral_concat":
            # Concatenate first and last token
            features = torch.cat([features[:, 0], features[:, -1]], dim=-1)  # (B, 2*D)
        elif self.encoder_pooling == "mean":
            # Mean pool all tokens
            features = features.mean(dim=1)  # (B, D)
        # else: no pooling, return (B, N, D)

        return features

    def forward(self, images, captions=None, mode='train'):
        """
        Forward pass.

        Args:
            images: (batch_size, 3, H, W)
            captions: List of caption strings (for training)
            mode: 'train' or 'generate'

        Returns:
            If mode='train': logits (batch_size, seq_len, vocab_size)
            If mode='generate': generated caption strings
        """
        # Encode image
        visual_features = self.encode_image(images)  # (B, D_enc)

        # Fuse visual features to decoder space
        visual_context = self.fusion(visual_features)  # (B, D_dec)

        if mode == 'train':
            assert captions is not None, "Captions required for training mode"

            # Prepare teacher forcing inputs
            tokenized = self.tokenizer.prepare_teacher_forcing_inputs(
                captions,
                device=images.device
            )

            # Decode with teacher forcing
            logits = self.decoder(
                tokenized['decoder_input_ids'],
                visual_context=visual_context
            )

            return {
                'logits': logits,
                'target_ids': tokenized['target_ids'],
                'attention_mask': tokenized['attention_mask']
            }

        elif mode == 'generate':
            # Autoregressive generation
            generated_ids = self.decoder.generate(
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                visual_context=visual_context,
                max_length=self.config.max_caption_length,
            )

            # Decode to text
            captions = self.tokenizer.decode(generated_ids)

            return {
                'captions': captions,
                'token_ids': generated_ids
            }

        else:
            raise ValueError(f"Invalid mode: {mode}")

    def generate_captions(self, images, temperature=1.0, top_k=None):
        """
        Generate captions for images.

        Args:
            images: (batch_size, 3, H, W)
            temperature: Sampling temperature
            top_k: Top-k sampling

        Returns:
            List of caption strings
        """
        self.eval()
        with torch.no_grad():
            visual_features = self.encode_image(images)
            visual_context = self.fusion(visual_features)

            generated_ids = self.decoder.generate(
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                visual_context=visual_context,
                max_length=self.config.max_caption_length,
                temperature=temperature,
                top_k=top_k,
            )

            captions = self.tokenizer.decode(generated_ids)

        return captions
