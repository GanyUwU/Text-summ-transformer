
import torch
from model import build_transformer
from tokenizer_utils import get_tokenizer
from pretrain_config import get_finetune_config
from train_summarization import greedy_decode
from datasets import load_dataset
from pathlib import Path

def inspect():
    config = get_finetune_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = get_tokenizer(config['tokenizer_model'])
    vocab_size = tokenizer.get_vocab_size()

    # Build model (ensure architecture matches exactly)
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

    # Load checkpoint
    ckpt_path = "weights_v11_nuclear/nuclear_summarizer_best.pt"
    if not Path(ckpt_path).exists():
        # Fallback to step checkpoint if best doesn't exist
        ckpt_path = "weights_v11_nuclear/nuclear_summarizer_step_1000.pt"
    
    print(f"Loading {ckpt_path}...")
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Failed to load with weights_only=False: {e}")
        checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load a few validation examples
    dataset = load_dataset(config['datasource'], config['dataset_version'], split='validation[:10]')
    
    for i, item in enumerate(dataset):
        article = item['article']
        ref = item['highlights']
        
        print(f"\n--- Example {i+1} ---")
        print(f"SOURCE (truncated): {article[:400]}...")
        
        # Tokenize
        src_tokens = tokenizer.encode(article)[:config['src_seq_len'] - 2]
        enc_input = torch.tensor([tokenizer.bos_id] + src_tokens + [tokenizer.eos_id]).unsqueeze(0).to(device)
        enc_mask = (enc_input != tokenizer.pad_id).view(1, 1, 1, -1)
        
        with torch.no_grad():
            # Encode
            enc_out = model.encode(enc_input, enc_mask)
            
            # Greedy Decode
            out_ids = greedy_decode(
                model, enc_out, enc_mask, enc_input,
                tokenizer, config['tgt_seq_len'], device,
                no_repeat_ngram=3
            )
            
            summary = tokenizer.decode(out_ids)
            print(f"REF: {ref[:150]}...")
            print(f"GEN: {summary}")
            
            # Check p_gen for first few steps
            # (We'd need to modify forward but let's just see output first)

if __name__ == "__main__":
    inspect()
