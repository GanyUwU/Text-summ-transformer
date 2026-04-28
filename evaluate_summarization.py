"""
Comprehensive evaluation script for summarization models.
Computes ROUGE scores on the full test set.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from model import build_transformer
from dataset_summarization import SummarizationDataset, causal_mask
from config_summarization import get_config, latest_weights_file_path
from datasets import load_dataset
from tokenizers import Tokenizer

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("ERROR: rouge_score not installed. Run: pip install rouge-score")
    exit(1)


def greedy_decode(model, source, source_mask, tokenizer, max_len, device):
    """Generate summary using greedy decoding"""
    sos_idx = tokenizer.token_to_id('[SOS]')
    eos_idx = tokenizer.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    
    while decoder_input.size(1) < max_len:
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        
        decoder_input = torch.cat([
            decoder_input, 
            torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)
        ], dim=1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def evaluate_model(config, model_path=None, num_samples=None):
    """
    Evaluate model on test set.
    
    Args:
        config: Configuration dictionary
        model_path: Path to model checkpoint (default: latest)
        num_samples: Number of samples to evaluate (default: all)
    """
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    device = torch.device(device)

    # Load tokenizer
    tokenizer_path = Path(config['tokenizer_file'].format('shared'))
    if not tokenizer_path.exists():
        print(f"ERROR: Tokenizer not found at {tokenizer_path}")
        print("Please train the model first to generate the tokenizer.")
        return
    
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"Tokenizer loaded: vocab size = {tokenizer.get_vocab_size()}")

    # Load test dataset
    print("Loading test dataset...")
    ds_test = load_dataset(
        config['datasource'], 
        config['dataset_version'], 
        split='test'
    )
    
    test_ds = SummarizationDataset(
        ds_test, 
        tokenizer, 
        config['src_seq_len'], 
        config['tgt_seq_len']
    )
    
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)
    print(f"Test samples: {len(test_ds)}")

    # Load model
    print("Loading model...")
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
    
    # Load checkpoint
    if model_path is None:
        model_path = latest_weights_file_path(config)
    
    if model_path is None:
        print("ERROR: No model checkpoint found!")
        return
    
    print(f"Loading checkpoint: {model_path}")
    state = torch.load(model_path)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    predictions = []
    references = []
    
    # Evaluate
    print("\nGenerating summaries...")
    count = 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            
            # Generate summary
            model_out = greedy_decode(
                model, 
                encoder_input, 
                encoder_mask, 
                tokenizer, 
                config['tgt_seq_len'], 
                device
            )
            
            # Decode text
            predicted_text = tokenizer.decode(model_out.detach().cpu().numpy())
            reference_text = batch["tgt_text"][0]
            
            predictions.append(predicted_text)
            references.append(reference_text)
            
            # Compute ROUGE
            scores = scorer.score(reference_text, predicted_text)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
            
            count += 1
            if num_samples and count >= num_samples:
                break
    
    # Compute average scores
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Samples evaluated: {len(rouge1_scores)}")
    print(f"\nROUGE-1: {avg_rouge1:.4f}")
    print(f"ROUGE-2: {avg_rouge2:.4f}")
    print(f"ROUGE-L: {avg_rougeL:.4f}")
    print("="*60)
    
    # Save results
    results_file = Path(config['datasource'] + '_' + config['model_folder']) / 'evaluation_results.txt'
    with open(results_file, 'w') as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Samples: {len(rouge1_scores)}\n")
        f.write(f"ROUGE-1: {avg_rouge1:.4f}\n")
        f.write(f"ROUGE-2: {avg_rouge2:.4f}\n")
        f.write(f"ROUGE-L: {avg_rougeL:.4f}\n")
    
    print(f"\nResults saved to: {results_file}")
    
    # Print some examples
    print("\n" + "="*60)
    print("EXAMPLE PREDICTIONS")
    print("="*60)
    for i in range(min(5, len(predictions))):
        print(f"\nExample {i+1}:")
        print(f"Reference: {references[i][:200]}...")
        print(f"Generated: {predictions[i][:200]}...")
        print("-"*60)
    
    return {
        'rouge1': avg_rouge1,
        'rouge2': avg_rouge2,
        'rougeL': avg_rougeL,
        'predictions': predictions,
        'references': references
    }


if __name__ == '__main__':
    config = get_config()
    
    # Evaluate on full test set (or specify num_samples for faster testing)
    # evaluate_model(config, num_samples=100)  # Quick test on 100 samples
    evaluate_model(config)  # Full evaluation