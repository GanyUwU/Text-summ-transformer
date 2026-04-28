"""
Demo script for generating summaries with trained model.
Usage: python inference_demo.py
"""

import torch
from pathlib import Path
from tokenizers import Tokenizer

from model import build_transformer
from config_summarization import get_config, latest_weights_file_path
from dataset_summarization import causal_mask


def load_model_and_tokenizer(config, device):
    """Load trained model and tokenizer"""
    # Load tokenizer
    tokenizer_path = Path(config['tokenizer_file'].format('shared'))
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    # Load model
    model = build_transformer(
        src_vocab_size=tokenizer.get_vocab_size(),
        tgt_vocab_size=tokenizer.get_vocab_size(),
        src_seq_len=config['src_seq_len'],
        tgt_seq_len=config['tgt_seq_len'],
        d_model=config['d_model'],
        N=config['num_layers'],
        h=config['num_heads'],
        dropout=config['dropout'],
        d_ff=config['d_ff']
    ).to(device)
    
    # Load weights
    model_path = latest_weights_file_path(config)
    if model_path is None:
        raise FileNotFoundError("No trained model found. Please train first.")
    
    print(f"Loading model from: {model_path}")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    
    return model, tokenizer


def preprocess_article(article, tokenizer, max_len):
    """Preprocess article for model input"""
    # Tokenize
    tokens = tokenizer.encode(article).ids
    
    # Truncate if needed
    if len(tokens) > max_len - 2:
        tokens = tokens[:max_len - 2]
    
    # Add special tokens
    sos_token = tokenizer.token_to_id("[SOS]")
    eos_token = tokenizer.token_to_id("[EOS]")
    pad_token = tokenizer.token_to_id("[PAD]")
    
    # Build input: [SOS] + tokens + [EOS] + padding
    input_tokens = [sos_token] + tokens + [eos_token]
    padding = [pad_token] * (max_len - len(input_tokens))
    input_tokens = input_tokens + padding
    
    # Convert to tensor
    encoder_input = torch.tensor(input_tokens, dtype=torch.int64).unsqueeze(0)
    encoder_mask = (encoder_input != pad_token).unsqueeze(0).unsqueeze(0).int()
    
    return encoder_input, encoder_mask


def generate_summary(model, article, tokenizer, config, device):
    """Generate summary for given article"""
    # Preprocess
    encoder_input, encoder_mask = preprocess_article(
        article, 
        tokenizer, 
        config['src_seq_len']
    )
    encoder_input = encoder_input.to(device)
    encoder_mask = encoder_mask.to(device)
    
    # Encode article
    encoder_output = model.encode(encoder_input, encoder_mask)
    
    # Initialize decoder
    sos_idx = tokenizer.token_to_id('[SOS]')
    eos_idx = tokenizer.token_to_id('[EOS]')
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(encoder_input).to(device)
    
    # Generate tokens
    with torch.no_grad():
        for _ in range(config['tgt_seq_len']):
            if decoder_input.size(1) >= config['tgt_seq_len']:
                break
            
            # Build mask
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)
            
            # Decode
            out = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            prob = model.project(out[:, -1])
            
            # Get next token
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat([
                decoder_input,
                torch.empty(1, 1).type_as(encoder_input).fill_(next_word.item()).to(device)
            ], dim=1)
            
            # Stop if EOS
            if next_word == eos_idx:
                break
    
    # Decode to text
    output_tokens = decoder_input.squeeze(0).detach().cpu().numpy()
    summary = tokenizer.decode(output_tokens)
    
    # Clean up special tokens
    summary = summary.replace('[SOS]', '').replace('[EOS]', '').replace('[PAD]', '').strip()
    
    return summary


def demo():
    """Interactive demo"""
    print("\n" + "="*80)
    print("TEXT SUMMARIZATION DEMO")
    print("="*80)
    
    # Setup
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        model, tokenizer = load_model_and_tokenizer(config, device)
        print("✓ Model loaded successfully")
        print(f"✓ Vocabulary size: {tokenizer.get_vocab_size()}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Example articles
    examples = [
        """
        Scientists have discovered a new species of deep-sea fish in the Mariana Trench.
        The fish, which glows in the dark, was found at a depth of 8,000 meters.
        Researchers say this discovery could help us understand how life adapts to extreme conditions.
        The fish has unique features including oversized eyes and bioluminescent organs.
        The team plans to conduct further studies to learn more about this fascinating creature.
        """,
        """
        A major tech company announced today that it will be releasing a new smartphone next month.
        The device features an advanced camera system with AI-powered photography capabilities.
        It also includes a faster processor and improved battery life compared to previous models.
        Pre-orders will begin next week, and the phone will be available in three colors.
        Analysts predict strong sales due to the innovative features and competitive pricing.
        """,
        """
        Climate scientists warn that global temperatures are rising faster than expected.
        New data shows that the past decade has been the warmest on record.
        Ice caps in polar regions are melting at an alarming rate, contributing to sea level rise.
        Experts urge immediate action to reduce carbon emissions and transition to renewable energy.
        Without significant changes, the consequences could be severe for future generations.
        """
    ]
    
    print("\n" + "="*80)
    print("EXAMPLE SUMMARIES")
    print("="*80)
    
    for i, article in enumerate(examples, 1):
        article = article.strip()
        print(f"\n{'='*80}")
        print(f"EXAMPLE {i}")
        print(f"{'='*80}")
        print(f"\nARTICLE:\n{article}")
        
        print("\nGenerating summary...")
        summary = generate_summary(model, article, tokenizer, config, device)
        
        print(f"\nGENERATED SUMMARY:\n{summary}")
    
    # Interactive mode
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("Enter your own articles to summarize (or 'quit' to exit)")
    
    while True:
        print("\n" + "-"*80)
        article = input("\nEnter article (or 'quit'): ").strip()
        
        if article.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not article:
            print("Please enter some text.")
            continue
        
        if len(article) < 20:
            print("Article too short. Please enter at least 20 characters.")
            continue
        
        print("\nGenerating summary...")
        try:
            summary = generate_summary(model, article, tokenizer, config, device)
            print(f"\nSUMMARY:\n{summary}")
        except Exception as e:
            print(f"Error generating summary: {e}")


if __name__ == '__main__':
    demo()