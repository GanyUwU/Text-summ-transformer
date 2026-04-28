import torch
from torch.utils.data import Dataset


class SummarizationDataset(Dataset):
    """
    Dataset for abstractive summarization using Transformer.
    Handles tokenization, padding, SOS/EOS, and mask creation.
    """

    def __init__(self, ds, tokenizer, src_seq_len, tgt_seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer = tokenizer
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len

        # Special token IDs
        self.sos_id = int(tokenizer.token_to_id("[SOS]"))
        self.eos_id = int(tokenizer.token_to_id("[EOS]"))
        self.pad_id = int(tokenizer.token_to_id("[PAD]"))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        # Support fallback keys for multi-dataset domains (CNN/DailyMail, XSum, SAMSum)
        src_text = item.get("article", item.get("document", item.get("dialogue", "")))
        tgt_text = item.get("highlights", item.get("summary", ""))

        # Tokenize article & summary
        enc_tokens = self.tokenizer.encode(src_text).ids
        dec_tokens = self.tokenizer.encode(tgt_text).ids

        # Truncate (leave room for SOS/EOS)
        enc_tokens = enc_tokens[: self.src_seq_len - 2]
        dec_tokens = dec_tokens[: self.tgt_seq_len - 1]

        # Calculate padding
        enc_pad = self.src_seq_len - len(enc_tokens) - 2
        dec_pad = self.tgt_seq_len - len(dec_tokens) - 1
        if enc_pad < 0 or dec_pad < 0:
            raise ValueError("Sequence length too short for tokenizer output.")

        # Build encoder input: [SOS] + article + [EOS] + [PAD]...
        encoder_input = torch.cat([
            torch.tensor([self.sos_id]),
            torch.tensor(enc_tokens),
            torch.tensor([self.eos_id]),
            torch.full((enc_pad,), self.pad_id)
        ], dim=0)

        # Build decoder input: [SOS] + summary + [PAD]...
        decoder_input = torch.cat([
            torch.tensor([self.sos_id]),
            torch.tensor(dec_tokens),
            torch.full((dec_pad,), self.pad_id)
        ], dim=0)

        # Build label: summary + [EOS] + [PAD]...
        label = torch.cat([
            torch.tensor(dec_tokens),
            torch.tensor([self.eos_id]),
            torch.full((dec_pad,), self.pad_id)
        ], dim=0)

        # Shape checks
        assert encoder_input.size(0) == self.src_seq_len
        assert decoder_input.size(0) == self.tgt_seq_len
        assert label.size(0) == self.tgt_seq_len

        # Build masks (boolean)
        # (1, 1, seq_len) -> Batched: (batch, 1, 1, seq_len)
        encoder_mask = (encoder_input != self.pad_id).unsqueeze(0).unsqueeze(1)
        
        # (1, 1, seq_len)
        decoder_pad_mask = (decoder_input != self.pad_id).unsqueeze(0).unsqueeze(1)
        
        # (1, seq_len, seq_len)
        causal = causal_mask(decoder_input.size(0))
        
        # (1, seq_len, seq_len) -> Batched: (batch, 1, seq_len, seq_len)
        decoder_mask = decoder_pad_mask & causal

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def causal_mask(size):
    """
    Creates a lower-triangular (causal) mask.
    Prevents attention to future tokens.
    Shape: (1, size, size), dtype=bool
    """
    mask = torch.tril(torch.ones((1, size, size), dtype=torch.bool))
    return mask
