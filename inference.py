"""
Inference script for the trained Transformer summarization model.

Supports both standard decoding and copy-aware decoding (pointer-generator).
When copy mechanism is enabled, the model can copy rare words (names, numbers)
directly from the source article instead of generating from vocabulary.
"""

import torch
from pathlib import Path

from model import build_transformer
from tokenizer_utils import get_tokenizer
from pretrain_config import get_finetune_config
from checkpoint_utils import load_checkpoint


def load_model(config, device):
    """Load the trained model from checkpoint."""
    # Load tokenizer (SentencePiece)
    tokenizer = get_tokenizer(config['tokenizer_model'])
    vocab_size = tokenizer.get_vocab_size()
    
    # Build model architecture (with same flags used during training)
    model = build_transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        src_seq_len=config['src_seq_len'],
        tgt_seq_len=config['tgt_seq_len'],
        d_model=config['d_model'],
        N=config['num_layers'],
        h=config['num_heads'],
        dropout=config['dropout'],
        d_ff=config['d_ff'],
        share_weights=config.get('share_weights', True),
        use_copy=config.get('use_copy', True),
    ).to(device)
    
    # Load trained weights
    model_folder = Path(config['model_folder'])
    best_path = model_folder / f"{config['model_basename']}best.pt"
    
    if best_path.exists():
        checkpoint_path = str(best_path)
    else:
        # Find latest checkpoint
        pattern = f"{config['model_basename']}*.pt"
        weight_files = sorted(model_folder.glob(pattern))
        if not weight_files:
            raise FileNotFoundError(f"No checkpoint found in {model_folder}! Train the model first.")
        checkpoint_path = str(weight_files[-1])
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode (disables dropout)
    model.eval()
    
    epoch = checkpoint.get('epoch', '?')
    rouge = checkpoint.get('rouge1', None)
    print(f"Model loaded (epoch {epoch}" + (f", ROUGE-1: {rouge:.4f})" if rouge else ")"))
    
    return model, tokenizer


def _causal_mask(size, device):
    """Create causal mask for autoregressive decoding — 4D (1,1,T,T)."""
    return torch.tril(torch.ones((1, 1, size, size), dtype=torch.bool, device=device))


def beam_search_decode(model, encoder_output, encoder_mask, encoder_input_ids, tokenizer, max_len, device, 
                       beam_size=4, no_repeat_ngram=3, length_penalty=0.6, min_len=10, 
                       temperature=1.0, repetition_penalty=1.2):
    """
    Robust Beam Search decoding with Copy Mechanism support and N-Gram blocking.
    """
    bos_id = tokenizer.bos_id
    eos_id = tokenizer.eos_id
    
    # Beam: (token_ids, score, finished)
    beams = [([bos_id], 0.0, False)]
    use_copy = model.copy_mechanism is not None
    
    for _ in range(max_len):
        new_candidates = []
        all_finished = True
        
        for tokens, score, finished in beams:
            if finished or tokens[-1] == eos_id:
                new_candidates.append((tokens, score, True))
                continue
            
            all_finished = False
            decoder_input = torch.tensor([tokens], dtype=torch.long).to(device)
            decoder_mask = _causal_mask(decoder_input.size(1), device)
            
            with torch.no_grad():
                if use_copy:
                    decoder_output, cross_attn = model.decode(
                        encoder_output, encoder_mask, decoder_input, decoder_mask, return_cross_attn=True
                    )
                    vocab_logits = model.project(decoder_output)
                    tgt_embed = model.tgt_pos(model.tgt_embed(decoder_input))
                    context_vector = torch.bmm(cross_attn, encoder_output)
                    final_dist, _ = model.copy_mechanism(
                        decoder_output, context_vector, tgt_embed, vocab_logits, cross_attn, encoder_input_ids
                    )
                    probs = final_dist[0, -1, :]
                else:
                    decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                    logits = model.project(decoder_output[0, -1, :])
                    
                    # Apply temperature scaling
                    if temperature != 1.0:
                        logits = logits / temperature
                        
                    probs = torch.softmax(logits, dim=-1)
            
            # Apply Repetition Penalty (only to previously generated tokens)
            if repetition_penalty != 1.0:
                for past_token in set(tokens):
                    # We apply penalty to probabilities (dividing them) to push them down
                    probs[past_token] /= repetition_penalty
            
            # N-Gram blocking
            if no_repeat_ngram > 0 and len(tokens) >= no_repeat_ngram:
                current_gram = tuple(tokens[-(no_repeat_ngram-1):])
                for i in range(len(tokens) - no_repeat_ngram + 1):
                    if tuple(tokens[i:i+no_repeat_ngram-1]) == current_gram:
                        probs[tokens[i+no_repeat_ngram-1]] = 1e-12
            
            # Force Min Length
            if len(tokens) < min_len:
                probs[eos_id] = 1e-12
                
            probs = probs / (probs.sum() + 1e-12)
            log_probs = torch.log(probs + 1e-12)
            
            # Top-k transitions
            topk_log_probs, topk_ids = torch.topk(log_probs, beam_size)
            
            for i in range(beam_size):
                next_id = topk_ids[i].item()
                next_score = score + topk_log_probs[i].item()
                new_candidates.append((tokens + [next_id], next_score, next_id == eos_id))
        
        if all_finished:
            break
            
        def get_score(cand):
            return cand[1] / (len(cand[0]) ** length_penalty)
            
        new_candidates.sort(key=get_score, reverse=True)
        beams = new_candidates[:beam_size]
        
    return beams[0][0]


def greedy_decode(model, encoder_output, encoder_mask, encoder_input_ids, tokenizer, max_len, device, 
                  no_repeat_ngram=3, repetition_penalty=1.2, min_len=10):
    """
    Greedy decoding with support for Copy Mechanism and N-Gram blocking.
    """
    bos_id = tokenizer.bos_id
    eos_id = tokenizer.eos_id
    
    decoder_input = torch.tensor([[bos_id]], dtype=torch.long).to(device)
    generated_tokens = []
    
    use_copy = model.copy_mechanism is not None
    
    for _ in range(max_len):
        decoder_mask = _causal_mask(decoder_input.size(1), device)
        
        with torch.no_grad():
            if use_copy:
                decoder_output, cross_attn = model.decode(
                    encoder_output, encoder_mask, decoder_input, decoder_mask,
                    return_cross_attn=True
                )
                vocab_logits = model.project(decoder_output)
                tgt_embed = model.tgt_pos(model.tgt_embed(decoder_input))
                context_vector = torch.bmm(cross_attn, encoder_output)
                
                final_dist, _ = model.copy_mechanism(
                    decoder_output, context_vector, tgt_embed,
                    vocab_logits, cross_attn, encoder_input_ids
                )
                probs = final_dist[0, -1, :]
            else:
                decoder_output = model.decode(
                    encoder_output, encoder_mask, decoder_input, decoder_mask
                )
                logits = model.project(decoder_output[0, -1, :])
                probs = torch.softmax(logits, dim=-1)
        
        # Repetition Penalty
        if repetition_penalty != 1.0:
            for token in set(generated_tokens):
                probs[token] /= repetition_penalty
            # Safety re-normalization
            probs = probs / (probs.sum() + 1e-12)

        # N-Gram blocking
        if no_repeat_ngram > 0 and len(generated_tokens) >= no_repeat_ngram - 1:
            current_gram = tuple(generated_tokens[-(no_repeat_ngram-1):])
            for i in range(len(generated_tokens) - no_repeat_ngram + 1):
                if tuple(generated_tokens[i:i+no_repeat_ngram-1]) == current_gram:
                    probs[generated_tokens[i+no_repeat_ngram-1]] = 0.0
        
        # Force Min Length
        if len(generated_tokens) < min_len:
            probs[eos_id] = 0.0
            
        next_token = torch.argmax(probs).item()
        if next_token == eos_id:
            break
            
        generated_tokens.append(next_token)
        decoder_input = torch.cat([decoder_input, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1)
        
    return [bos_id] + generated_tokens


def summarize(model, tokenizer, article_text, config, device, max_length=128, min_length=30, 
              no_repeat_ngram_size=3, repetition_penalty=1.2, temperature=1.0, 
              beam_size=1, length_penalty=0.6):
    """
    Generate a summary for the given article text with optional Beam Search.
    """
    if max_length is None:
        max_length = config['tgt_seq_len']
    
    # Tokenization
    src_tokens = tokenizer.encode(article_text)[:config['src_seq_len'] - 2]
    enc_input = [tokenizer.bos_id] + src_tokens + [tokenizer.eos_id]
    enc_padding = config['src_seq_len'] - len(enc_input)
    enc_input = enc_input + [tokenizer.pad_id] * enc_padding
    
    encoder_input = torch.tensor(enc_input, dtype=torch.long).unsqueeze(0).to(device)
    encoder_mask = (encoder_input != tokenizer.pad_id).unsqueeze(1).unsqueeze(1).to(device)
    
    with torch.no_grad():
        encoder_output = model.encode(encoder_input, encoder_mask)
        
        if beam_size > 1:
            out_ids = beam_search_decode(
                model, encoder_output, encoder_mask, encoder_input,
                tokenizer, max_length, device,
                beam_size=beam_size,
                no_repeat_ngram=no_repeat_ngram_size,
                length_penalty=length_penalty,
                min_len=min_length,
                temperature=temperature,
                repetition_penalty=repetition_penalty
            )
        else:
            out_ids = greedy_decode(
                model, encoder_output, encoder_mask, encoder_input,
                tokenizer, max_length, device,
                no_repeat_ngram=no_repeat_ngram_size,
                repetition_penalty=repetition_penalty,
                min_len=min_length
            )
    
    return tokenizer.decode(out_ids)


def main():
    """Interactive summarization demo."""
    config = get_finetune_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model, tokenizer = load_model(config, device)
    
    has_copy = model.copy_mechanism is not None
    print(f"\nCopy mechanism: {'enabled' if has_copy else 'disabled'}")
    print(f"Weight sharing: {'enabled' if model.src_embed is model.tgt_embed else 'disabled'}")
    
    print("\n" + "="*60)
    print("TRANSFORMER SUMMARIZATION MODEL")
    print("="*60)
    print("Enter an article to summarize (or 'quit' to exit)")
    print("Tip: Paste a news article for best results\n")
    
    while True:
        print("-" * 60)
        article = input("Article: ").strip()
        
        if article.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if len(article) < 50:
            print("Please enter a longer article (at least 50 characters)")
            continue
        
        print("\nGenerating summary...")
        summary = summarize(model, tokenizer, article, config, device)
        print(f"\nSUMMARY: {summary}\n")


def demo():
    """Run a quick demo with a sample article."""
    config = get_finetune_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, tokenizer = load_model(config, device)
    
    sample_article = """
    Scientists have discovered a new species of deep-sea fish in the Pacific Ocean. 
    The fish, which lives at depths of over 8,000 meters, has unique adaptations that 
    allow it to survive the extreme pressure. Researchers from the University of Tokyo 
    used remotely operated vehicles to capture footage of the creature. The discovery 
    adds to our understanding of life in the deepest parts of the ocean.
    """
    
    print("\n" + "="*60)
    print("SAMPLE ARTICLE:")
    print(sample_article.strip())
    print("\n" + "="*60)
    
    summary = summarize(model, tokenizer, sample_article, config, device)
    print(f"GENERATED SUMMARY: {summary}")


if __name__ == '__main__':
    main()
