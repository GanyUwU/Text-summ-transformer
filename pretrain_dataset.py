"""
Pre-training Dataset with BART-style Denoising

This module implements text corruption techniques that teach the model:
1. Token meanings (via masking)
2. Grammar and word order (via deletion)
3. Document structure (via sentence shuffling)
"""

import torch
from torch.utils.data import Dataset
import random
import re


class DenoisingDataset(Dataset):
    """
    Dataset for denoising auto-encoding pre-training.
    
    Takes clean text, corrupts it, and trains the model to reconstruct
    the original text.
    """
    
    def __init__(self, texts, tokenizer, seq_len, config):
        """
        Args:
            texts: List of text strings (articles)
            tokenizer: Tokenizer object
            seq_len: Maximum sequence length
            config: Pre-training configuration dict
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.config = config
        
        # Get special token IDs
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.sos_id = tokenizer.token_to_id("[SOS]")
        self.eos_id = tokenizer.token_to_id("[EOS]")
        self.unk_id = tokenizer.token_to_id("[UNK]")
        
        # Add MASK token if not exists
        mask_id = tokenizer.token_to_id("[MASK]")
        if mask_id is None:
            # Use UNK as mask if MASK doesn't exist
            self.mask_id = self.unk_id
        else:
            self.mask_id = mask_id
        
        # Noise parameters
        self.mask_prob = config.get('mask_prob', 0.15)
        self.delete_prob = config.get('delete_prob', 0.10)
        self.shuffle_sentences = config.get('shuffle_sentences', True)
    
    def __len__(self):
        return len(self.texts)
    
    def add_noise(self, tokens):
        """
        Apply BART-style noise to token sequence.
        
        Noise types:
        1. Token masking: Replace random tokens with [MASK]
        2. Token deletion: Remove random tokens entirely
        
        Args:
            tokens: List of token IDs
        
        Returns:
            Corrupted token list
        """
        if len(tokens) == 0:
            return tokens
        
        corrupted = []
        
        for token in tokens:
            # Skip special tokens
            if token in [self.pad_id, self.sos_id, self.eos_id]:
                corrupted.append(token)
                continue
            
            rand = random.random()
            
            if rand < self.delete_prob:
                # Delete this token (don't add to output)
                continue
            elif rand < self.delete_prob + self.mask_prob:
                # Mask this token
                corrupted.append(self.mask_id)
            else:
                # Keep original token
                corrupted.append(token)
        
        return corrupted
    
    def shuffle_sentence_order(self, text):
        """
        Shuffle the order of sentences in the text.
        
        This teaches the model about document structure and
        coherence.
        """
        if not self.shuffle_sentences:
            return text
        
        # Split into sentences (simple approach)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if len(sentences) > 1:
            random.shuffle(sentences)
        
        return ' '.join(sentences)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Step 1: Shuffle sentences (before tokenization)
        shuffled_text = self.shuffle_sentence_order(text)
        
        # Step 2: Tokenize
        original_tokens = self.tokenizer.encode(text).ids
        corrupted_tokens = self.tokenizer.encode(shuffled_text).ids
        
        # Step 3: Apply token-level noise to corrupted version
        corrupted_tokens = self.add_noise(corrupted_tokens)
        
        # Step 4: Truncate to fit sequence length
        # Leave room for SOS/EOS
        max_tokens = self.seq_len - 2
        original_tokens = original_tokens[:max_tokens]
        corrupted_tokens = corrupted_tokens[:max_tokens]
        
        # Step 5: Build encoder input (corrupted): [SOS] + corrupted + [EOS] + [PAD]
        enc_input = [self.sos_id] + corrupted_tokens + [self.eos_id]
        enc_padding = self.seq_len - len(enc_input)
        if enc_padding < 0:
            enc_input = enc_input[:self.seq_len]
            enc_padding = 0
        enc_input = enc_input + [self.pad_id] * enc_padding
        
        # Step 6: Build decoder input (original shifted): [SOS] + original + [PAD]
        dec_input = [self.sos_id] + original_tokens
        dec_padding = self.seq_len - len(dec_input)
        if dec_padding < 0:
            dec_input = dec_input[:self.seq_len]
            dec_padding = 0
        dec_input = dec_input + [self.pad_id] * dec_padding
        
        # Step 7: Build label (original): original + [EOS] + [PAD]
        label = original_tokens + [self.eos_id]
        label_padding = self.seq_len - len(label)
        if label_padding < 0:
            label = label[:self.seq_len]
            label_padding = 0
        label = label + [self.pad_id] * label_padding
        
        # Convert to tensors
        encoder_input = torch.tensor(enc_input, dtype=torch.long)
        decoder_input = torch.tensor(dec_input, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        
        # Build masks
        encoder_mask = (encoder_input != self.pad_id).unsqueeze(0).unsqueeze(0)
        decoder_mask = (decoder_input != self.pad_id).unsqueeze(0) & causal_mask(self.seq_len)
        
        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_mask': encoder_mask,
            'decoder_mask': decoder_mask,
            'label': label,
        }


def causal_mask(size):
    """
    Creates a lower-triangular (causal) mask.
    Prevents attention to future tokens.
    """
    mask = torch.tril(torch.ones((1, size, size), dtype=torch.bool))
    return mask


# Quick test
if __name__ == '__main__':
    from tokenizers import Tokenizer
    from pretrain_config import get_pretrain_config
    
    config = get_pretrain_config()
    
    # Mock tokenizer test
    print("DenoisingDataset module loaded successfully!")
    print(f"Mask probability: {config['mask_prob']}")
    print(f"Delete probability: {config['delete_prob']}")
    print(f"Shuffle sentences: {config['shuffle_sentences']}")
