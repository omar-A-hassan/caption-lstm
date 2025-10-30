import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional
from transformers import BertModel

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
    vocab_size: int = 30523  # BERT vocab size (30522) + [EOS] token (1)
    decoder_dim: int = 512
    decoder_num_blocks: int = 3
    decoder_num_heads: int = 4
    decoder_dropout: float = 0.2
    max_caption_length: int = 50

    # Tokenizer
    tokenizer_model: str = "bert-base-uncased"

    # Alignment loss (optional)
    alignment_loss_weight: float = 0.0
    alignment_loss_type: str = "geom"
    alignment_loss_kwargs: Optional[Dict] = None
    alignment_normalize: bool = True
    alignment_proj_dim: Optional[int] = None


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

        # Projection heads for alignment objective
        alignment_dim = config.alignment_proj_dim or config.decoder_dim
        self.alignment_dim = alignment_dim
        self.visual_align_proj = nn.Linear(
            config.encoder_dim,
            alignment_dim,
        )
        self.decoder_align_proj = nn.Linear(
            config.decoder_dim,
            alignment_dim,
            bias=False,
        )

        self.alignment_loss_weight = config.alignment_loss_weight
        self.alignment_loss_fn = None
        if self.alignment_loss_weight > 0.0:
            if config.alignment_loss_type.lower() != "geom":
                raise ValueError(
                    f"Unsupported alignment loss type: {config.alignment_loss_type}. "
                    "Only 'geom' is currently implemented."
                )
            try:
                from geomloss import SamplesLoss
            except ImportError as exc:
                raise ImportError(
                    "geomloss is required for alignment_loss_weight > 0. "
                    "Install it via `pip install geomloss`."
                ) from exc

            default_kwargs = dict(loss="sinkhorn", p=2, blur=0.05, scaling=0.9, backend="auto")
            loss_kwargs = default_kwargs
            if config.alignment_loss_kwargs is not None:
                loss_kwargs = {**default_kwargs, **config.alignment_loss_kwargs}
            self.alignment_loss_fn = SamplesLoss(**loss_kwargs)

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

        # Initialize decoder embeddings with pretrained BERT embeddings
        self._init_pretrained_embeddings()

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

    def _init_pretrained_embeddings(self):
        """
        Initialize decoder token embeddings with pretrained BERT embeddings.

        This provides a strong initialization for word representations, leveraging
        BERT's knowledge from pretraining on billions of tokens. The BiLSTM paper
        (Fig 7) uses an "Embedding" layer which typically refers to pretrained
        embeddings that are fine-tuned during training.

        Key benefits:
        - Faster convergence (5-10 epochs vs 50+ epochs)
        - Better performance with limited data
        - Improved handling of rare words
        - Semantically meaningful initial word vectors
        """
        try:
            print("Loading pretrained BERT embeddings...")

            # Load BERT model to extract embeddings
            bert_model = BertModel.from_pretrained(self.config.tokenizer_model)
            bert_embeddings = bert_model.embeddings.word_embeddings.weight.data

            # BERT embeddings are 768-dim, but our decoder might use different dim
            bert_dim = bert_embeddings.shape[1]
            decoder_dim = self.decoder.config.embedding_dim
            vocab_size = self.decoder.config.vocab_size

            print(f"  BERT embedding dim: {bert_dim}")
            print(f"  Decoder embedding dim: {decoder_dim}")
            print(f"  Vocab size: {vocab_size}")

            if bert_dim == decoder_dim:
                # Same dimension - direct copy
                # Copy embeddings for original BERT tokens
                original_bert_vocab = min(bert_embeddings.shape[0], vocab_size)
                self.decoder.token_embedding.weight.data[:original_bert_vocab].copy_(
                    bert_embeddings[:original_bert_vocab]
                )
                print(f"  ✓ Copied {original_bert_vocab} pretrained embeddings directly")

                # New tokens (like [EOS]) keep random initialization
                if vocab_size > original_bert_vocab:
                    num_new = vocab_size - original_bert_vocab
                    print(f"  ✓ {num_new} new tokens keep random initialization")

            else:
                # Different dimension - use linear projection
                print(f"  Using projection: {bert_dim} -> {decoder_dim}")

                # Create a simple linear projection
                projection = nn.Linear(bert_dim, decoder_dim, bias=False)
                nn.init.xavier_uniform_(projection.weight)

                # Project BERT embeddings
                with torch.no_grad():
                    original_bert_vocab = min(bert_embeddings.shape[0], vocab_size)
                    projected_embeddings = projection(bert_embeddings[:original_bert_vocab])
                    self.decoder.token_embedding.weight.data[:original_bert_vocab].copy_(
                        projected_embeddings
                    )

                print(f"  ✓ Projected and copied {original_bert_vocab} embeddings")

                if vocab_size > original_bert_vocab:
                    num_new = vocab_size - original_bert_vocab
                    print(f"  ✓ {num_new} new tokens keep random initialization")

            # Clean up BERT model to save memory
            del bert_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("✓ Pretrained embeddings initialized successfully!")

        except Exception as e:
            print(f"⚠ Warning: Could not load pretrained embeddings: {e}")
            print("  Continuing with random initialization...")

    def encode_image(self, images, return_tokens: bool = False):
        """
        Encode images to visual features.

        Args:
            images: (batch_size, 3, H, W)
            return_tokens: Whether to also return patch/token-level features

        Returns:
            visual_features: (batch_size, encoder_dim) or (batch_size, N, encoder_dim)
            (optionally) encoder_tokens: (batch_size, N, encoder_dim)
        """
        # Encoder returns (B, N, D) where N = num_patches
        features = self.encoder(images)

        pooled = self._apply_pooling(features)

        if return_tokens:
            return pooled, features
        return pooled

    def _apply_pooling(self, token_features: torch.Tensor) -> torch.Tensor:
        """Apply the configured pooling on encoder token features."""
        if self.encoder_pooling == "bilateral_avg":
            return (token_features[:, 0] + token_features[:, -1]) / 2
        if self.encoder_pooling == "bilateral_concat":
            return torch.cat([token_features[:, 0], token_features[:, -1]], dim=-1)
        if self.encoder_pooling == "mean":
            return token_features.mean(dim=1)
        # No pooling -> return token grid
        return token_features

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
        visual_features, encoder_tokens = self.encode_image(images, return_tokens=True)  # pooled + tokens

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
            decoder_output = self.decoder(
                tokenized['decoder_input_ids'],
                visual_context=visual_context,
                return_hidden_states=self.alignment_loss_weight > 0.0,
            )
            logits = decoder_output["logits"]
            hidden_states = decoder_output.get("hidden_states")

            output = {
                'logits': logits,
                'target_ids': tokenized['target_ids'],
                'attention_mask': tokenized['attention_mask']
            }

            if self.alignment_loss_weight > 0.0 and hidden_states is not None:
                alignment_loss = self.compute_alignment_loss(
                    encoder_tokens=encoder_tokens,
                    decoder_hidden=hidden_states,
                    attention_mask=tokenized['attention_mask'],
                )
                output['alignment_loss'] = alignment_loss

            return output

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

    def compute_alignment_loss(self, encoder_tokens, decoder_hidden, attention_mask):
        """Compute visual-linguistic alignment loss using geomloss."""
        if self.alignment_loss_fn is None:
            raise RuntimeError("Alignment loss requested but alignment_loss_fn is not initialized.")

        # Project encoder and decoder tokens into shared alignment space
        visual_embed = self.visual_align_proj(encoder_tokens)  # (B, N, alignment_dim)
        decoder_embed = self.decoder_align_proj(decoder_hidden)  # (B, S, alignment_dim)

        if self.config.alignment_normalize:
            visual_embed = F.normalize(visual_embed, dim=-1)
            decoder_embed = F.normalize(decoder_embed, dim=-1)

        batch_size, num_patches, _ = visual_embed.shape

        # Uniform weights for visual tokens
        visual_weights = visual_embed.new_full((batch_size, num_patches), 1.0 / num_patches)

        # Attention mask -> weights (avoid division by zero)
        text_mask = attention_mask.float()
        text_weights = text_mask / text_mask.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        decoder_embed = decoder_embed * text_mask.unsqueeze(-1)

        loss = self.alignment_loss_fn(
            visual_embed,
            decoder_embed,
            visual_weights,
            text_weights,
        )

        return loss * self.alignment_loss_weight
