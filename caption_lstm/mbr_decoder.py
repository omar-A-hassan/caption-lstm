"""
MBR Decoding wrapper for image captioning using naist-nlp/mbrs library.

Install: pip install mbrs
Repo: https://github.com/naist-nlp/mbrs
"""

import torch
from typing import List, Optional


class MBRCaptionDecoder:
    """
    Minimum Bayes Risk (MBR) Decoding for image captioning using mbrs library.

    Generates N caption candidates via sampling, then uses MBR decoding
    with a learned metric (COMET, BERTScore, etc.) to select the best caption.
    """

    def __init__(
        self,
        model,
        num_candidates: int = 16,
        metric_name: str = "comet",
        metric_model: str = "Unbabel/wmt22-comet-da",
        batch_size: int = 64,
        fp16: bool = True,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ):
        """
        Args:
            model: ViLCap model instance
            num_candidates: Number of caption candidates to generate per image
            metric_name: Metric to use ('comet', 'bleurt', 'bertscore', 'bleu', 'chrf')
            metric_model: Pretrained model for the metric (for COMET, BLEURT, etc.)
            batch_size: Batch size for metric computation
            fp16: Use FP16 for metric computation (faster)
            temperature: Sampling temperature for caption generation
            top_k: Top-k sampling parameter
        """
        try:
            from mbrs.metrics import MetricCOMET, MetricBLEURT, MetricBERTScore
            from mbrs.decoders import DecoderMBR
            self.mbrs_available = True
        except ImportError:
            raise ImportError(
                "mbrs library not found. Install with: pip install mbrs\n"
                "Repo: https://github.com/naist-nlp/mbrs"
            )

        self.model = model
        self.num_candidates = num_candidates
        self.temperature = temperature
        self.top_k = top_k

        # Initialize metric
        if metric_name.lower() == "comet":
            from mbrs.metrics import MetricCOMET
            metric_cfg = MetricCOMET.Config(
                model=metric_model,
                batch_size=batch_size,
                fp16=fp16,
            )
            self.metric = MetricCOMET(metric_cfg)

        elif metric_name.lower() == "bleurt":
            from mbrs.metrics import MetricBLEURT
            metric_cfg = MetricBLEURT.Config(
                model=metric_model,
                batch_size=batch_size,
                fp16=fp16,
            )
            self.metric = MetricBLEURT(metric_cfg)

        elif metric_name.lower() == "bertscore":
            from mbrs.metrics import MetricBERTScore
            metric_cfg = MetricBERTScore.Config(
                model=metric_model,
                batch_size=batch_size,
            )
            self.metric = MetricBERTScore(metric_cfg)

        else:
            raise ValueError(
                f"Unsupported metric: {metric_name}. "
                "Supported: 'comet', 'bleurt', 'bertscore'"
            )

        # Initialize MBR decoder
        from mbrs.decoders import DecoderMBR
        decoder_cfg = DecoderMBR.Config()
        self.mbr_decoder = DecoderMBR(decoder_cfg, self.metric)

    @torch.no_grad()
    def decode(self, images: torch.Tensor, source_text: Optional[List[str]] = None) -> List[str]:
        """
        Generate captions using MBR decoding.

        Args:
            images: (batch_size, 3, H, W) - Input images
            source_text: Optional source text (e.g., "image description")
                        If None, uses empty string

        Returns:
            List of selected captions (one per image)
        """
        self.model.eval()
        batch_size = images.shape[0]

        # Default source text if not provided
        if source_text is None:
            source_text = [""] * batch_size

        # Encode images once
        visual_features = self.model.encode_image(images)  # (B, D_enc) or (B, N, D_enc)
        visual_context = self.model.fusion(visual_features)  # (B, D_dec) or (B, N, D_dec)

        # Generate candidates and perform MBR for each image
        selected_captions = []

        for b in range(batch_size):
            # Get visual context for this image
            visual_ctx_single = visual_context[b:b+1]  # (1, D) or (1, N, D)

            # Expand visual context for N candidates
            # Handle both 2D (B, D) and 3D (B, N, D) cases
            if visual_ctx_single.dim() == 2:
                # (1, D) -> (num_candidates, D)
                visual_ctx_batch = visual_ctx_single.expand(self.num_candidates, -1)
            else:
                # (1, N, D) -> (num_candidates, N, D)
                visual_ctx_batch = visual_ctx_single.expand(self.num_candidates, -1, -1)

            # Generate N candidates via sampling
            generated_ids = self.model.decoder.generate(
                bos_token_id=self.model.tokenizer.bos_token_id,
                eos_token_id=self.model.tokenizer.eos_token_id,
                pad_token_id=self.model.tokenizer.pad_token_id,
                visual_context=visual_ctx_batch,
                max_length=self.model.config.max_caption_length,
                temperature=self.temperature,
                top_k=self.top_k,
            )

            # Decode to text
            candidates = self.model.tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Perform MBR decoding
            # In mbrs, we use candidates as both hypotheses and pseudo-references
            output = self.mbr_decoder.decode(
                hypotheses=candidates,
                references=candidates,  # Self-referencing for MBR
                source=source_text[b],
                nbest=1
            )

            selected_captions.append(output.sentence)

        return selected_captions

    @torch.no_grad()
    def decode_batch(self, images: torch.Tensor, source_text: Optional[List[str]] = None) -> List[str]:
        """
        Batched version of decode for efficiency.

        Args:
            images: (batch_size, 3, H, W)
            source_text: Optional list of source texts

        Returns:
            List of selected captions
        """
        # For now, just call decode (which processes one image at a time)
        # Can be optimized later to process multiple images in parallel
        return self.decode(images, source_text)
