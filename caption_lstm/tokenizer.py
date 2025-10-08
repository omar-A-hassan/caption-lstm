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
        # Encode captions first WITHOUT special tokens
        encoded = self.tokenizer(
            captions,
            padding=False,  # Don't pad yet
            truncation=True,
            max_length=self.max_length - 1,  # Leave room for EOS
            add_special_tokens=False,  # Don't add [CLS] and [SEP]
            return_tensors='pt'
        )

        # Manually append EOS token ID to each sequence
        # This ensures captions end with actual [EOS] token, not tokenized text
        input_ids = encoded['input_ids']
        batch_size = input_ids.shape[0]

        # Append EOS token to each sequence
        eos_tokens = torch.full((batch_size, 1), self.eos_token_id, dtype=torch.long)
        input_ids = torch.cat([input_ids, eos_tokens], dim=1)

        # Now pad to max_length
        seq_len = input_ids.shape[1]
        if seq_len < self.max_length:
            padding = torch.full(
                (batch_size, self.max_length - seq_len),
                self.pad_token_id,
                dtype=torch.long
            )
            input_ids = torch.cat([input_ids, padding], dim=1)

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != self.pad_token_id).long()

        encoded = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

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
