import torch
from transformers import AutoTokenizer


class CaptionTokenizer:
    """Wrapper for HuggingFace tokenizers for image captioning."""

    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 50):
        """
        Args:
            model_name: HuggingFace model name for tokenizer
            max_length: Maximum caption length
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

        # Add EOS token for proper sequence termination
        # Note: BERT already has [PAD], [CLS], [SEP], [UNK], [MASK]
        # We add [EOS] specifically for generation stopping criterion
        special_tokens = {'eos_token': '[EOS]'}
        num_added = self.tokenizer.add_special_tokens(special_tokens)

        # Use BERT's CLS as our BOS (beginning of sequence) token
        # This is semantically reasonable: CLS marks "start of classification input"
        # which is similar to "start of generation output"
        self.bos_token_id = self.tokenizer.cls_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.vocab_size = len(self.tokenizer)

    def encode(self, captions, device=None):
        """
        Encode captions to token IDs with proper EOS token.

        Args:
            captions: List of caption strings
            device: Target device for tensors

        Returns:
            dict with 'input_ids' and 'attention_mask'

        Note:
            BERT tokenizer by default adds [CLS] at start and [SEP] at end.
            We disable default special tokens and manually add our [EOS] token
            to ensure proper sequence termination during generation.
        """
        # Manually append EOS token to each caption
        # This ensures captions end with [EOS] instead of BERT's default [SEP]
        captions_with_eos = [f"{cap} [EOS]" for cap in captions]

        # Encode WITHOUT automatic special token addition
        # This prevents BERT from adding [CLS] and [SEP]
        encoded = self.tokenizer(
            captions_with_eos,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,  # Don't add [CLS] and [SEP]
            return_tensors='pt'
        )

        if device is not None:
            encoded = {k: v.to(device) for k, v in encoded.items()}

        return encoded

    def decode(self, token_ids, skip_special_tokens=True):
        """
        Decode token IDs to caption strings.

        Args:
            token_ids: Tensor of token IDs (batch_size, seq_len)
            skip_special_tokens: Whether to skip special tokens

        Returns:
            List of caption strings
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()

        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)

    def prepare_teacher_forcing_inputs(self, captions, device=None):
        """
        Prepare inputs for teacher forcing during training.

        Args:
            captions: List of caption strings
            device: Target device

        Returns:
            dict with 'decoder_input_ids' (with BOS) and 'target_ids' (with EOS)
        """
        # Encode captions
        encoded = self.encode(captions, device=device)
        input_ids = encoded['input_ids']

        # Prepend BOS token and remove last token for decoder input
        bos_tokens = torch.full((input_ids.size(0), 1), self.bos_token_id,
                                dtype=input_ids.dtype, device=input_ids.device)
        decoder_input_ids = torch.cat([bos_tokens, input_ids[:, :-1]], dim=1)

        # Target is the original sequence (which should end with EOS/PAD)
        target_ids = input_ids

        return {
            'decoder_input_ids': decoder_input_ids,
            'target_ids': target_ids,
            'attention_mask': encoded['attention_mask']
        }
