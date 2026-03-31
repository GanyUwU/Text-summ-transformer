"""
SentencePiece Tokenizer Utilities

Professional tokenizer using SentencePiece BPE/Unigram model
for robust subword tokenization.
"""

import sentencepiece as spm
from pathlib import Path
from datasets import load_dataset
import os


def train_tokenizer_on_wikipedia(
    vocab_size: int = 32000,
    model_type: str = "unigram",  # "bpe" or "unigram"
    output_prefix: str = "tokenizer_sp",
    sample_size: int = 1000000,  # Number of sentences to sample
):
    """
    Train a SentencePiece tokenizer on Wikipedia.
    
    Args:
        vocab_size: Vocabulary size (32k recommended)
        model_type: "bpe" or "unigram" (unigram recommended for summarization)
        output_prefix: Output file prefix
        sample_size: Number of sentences to train on
    """
    print("="*60)
    print("TRAINING SENTENCEPIECE TOKENIZER")
    print("="*60)
    
    # Check if tokenizer already exists
    if Path(f"{output_prefix}.model").exists():
        print(f"✓ Tokenizer already exists: {output_prefix}.model")
        return f"{output_prefix}.model"
    
    # Use Wikipedia (wikimedia format - works without trust_remote_code)
    print("\n1. Loading Wikipedia (streaming)...")
    wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    
    # Create temp file with training text
    train_file = "tokenizer_train_data.txt"
    print(f"\n2. Extracting {sample_size:,} sentences to {train_file}...")
    
    with open(train_file, 'w', encoding='utf-8') as f:
        count = 0
        for article in wiki:
            # Handle different dataset formats
            if 'text' in article:
                text = article['text']
            elif 'article' in article:
                text = article['article']
            else:
                text = str(article)
            
            # Split into sentences and write
            sentences = text.replace('\n', ' ').split('. ')
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 20:  # Skip very short sentences
                    f.write(sent + '\n')
                    count += 1
                    if count >= sample_size:
                        break
            if count >= sample_size:
                break
            if count % 100000 == 0 and count > 0:
                print(f"   Extracted {count:,} sentences...")
    
    print(f"   Done! Extracted {count:,} sentences")
    
    # Train SentencePiece
    print(f"\n3. Training SentencePiece ({model_type}, vocab={vocab_size})...")
    
    spm.SentencePieceTrainer.train(
        input=train_file,
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=0.9995,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece='[PAD]',
        unk_piece='[UNK]',
        bos_piece='[BOS]',
        eos_piece='[EOS]',
        user_defined_symbols=['[MASK]'],
        num_threads=os.cpu_count(),
        train_extremely_large_corpus=True,
    )
    
    print(f"✓ Tokenizer saved: {output_prefix}.model")
    
    # Cleanup
    if Path(train_file).exists():
        os.remove(train_file)
        print(f"   Cleaned up {train_file}")
    
    return f"{output_prefix}.model"


class SentencePieceTokenizer:
    """
    Wrapper for SentencePiece tokenizer with interface matching HuggingFace tokenizers.
    """
    
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        
        # Special token IDs
        self.pad_id = self.sp.pad_id()
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        
        # MASK token
        self.mask_id = self.sp.piece_to_id('[MASK]')
        if self.mask_id == self.unk_id:
            # Fallback if MASK not found
            self.mask_id = self.unk_id
    
    def encode(self, text: str, add_special_tokens: bool = False) -> list:
        """Encode text to token IDs."""
        ids = self.sp.encode(text, out_type=int)
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids
    
    def decode(self, ids: list) -> str:
        """Decode token IDs to text."""
        # Filter special tokens for cleaner output
        filtered = [i for i in ids if i not in [self.pad_id, self.bos_id, self.eos_id]]
        return self.sp.decode(filtered)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.sp.get_piece_size()
    
    def token_to_id(self, token: str) -> int:
        """Get ID for a token."""
        return self.sp.piece_to_id(token)
    
    def id_to_token(self, id: int) -> str:
        """Get token for an ID."""
        return self.sp.id_to_piece(id)
    
    def __len__(self):
        return self.get_vocab_size()


def get_tokenizer(model_path: str = "tokenizer_sp.model") -> SentencePieceTokenizer:
    """Load or train tokenizer."""
    if not Path(model_path).exists():
        print(f"Tokenizer not found at {model_path}")
        print("Training new tokenizer...")
        train_tokenizer_on_wikipedia(output_prefix=model_path.replace('.model', ''))
    
    return SentencePieceTokenizer(model_path)


if __name__ == '__main__':
    # Train tokenizer if run directly
    model_path = train_tokenizer_on_wikipedia(
        vocab_size=32000,
        model_type="unigram",
        sample_size=500000  # 500k sentences for reasonable training
    )
    
    # Test it
    print("\n" + "="*60)
    print("TESTING TOKENIZER")
    print("="*60)
    
    tokenizer = SentencePieceTokenizer(model_path)
    print(f"\nVocab size: {tokenizer.get_vocab_size()}")
    
    test_text = "The quick brown fox jumps over the lazy dog."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"\nOriginal: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    print(f"\nSpecial tokens:")
    print(f"  PAD: {tokenizer.pad_id}")
    print(f"  UNK: {tokenizer.unk_id}")
    print(f"  BOS: {tokenizer.bos_id}")
    print(f"  EOS: {tokenizer.eos_id}")
    print(f"  MASK: {tokenizer.mask_id}")
