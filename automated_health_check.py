
import torch
import numpy as np
from model import build_transformer
from tokenizer_utils import get_tokenizer
from pretrain_config import get_finetune_config
from pathlib import Path

def health_check():
    config = get_finetune_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = get_tokenizer(config['tokenizer_model'])
    vocab_size = tokenizer.get_vocab_size()

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

    ckpt_path = "weights_v11_nuclear/nuclear_summarizer_step_12000.pt"
    if not Path(ckpt_path).exists():
        print(f"File not found: {ckpt_path}")
        return

    print(f"Analyzing {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Test with a specific input
    input_text = "The Federal Bureau of Investigation is leading the search for Cayman Naib, a 13-year-old 8th grade student who has been missing since Wednesday."
    src_tokens = tokenizer.encode(input_text)[:config['src_seq_len'] - 2]
    enc_input = torch.tensor([tokenizer.bos_id] + src_tokens + [tokenizer.eos_id]).unsqueeze(0).to(device)
    enc_mask = (enc_input != tokenizer.pad_id).view(1, 1, 1, -1)
    
    # We'll use a dummy decoder input [BOS]
    dec_input = torch.tensor([[tokenizer.bos_id]]).to(device)
    dec_mask = torch.ones((1, 1, 1, 1), dtype=torch.bool).to(device)

    with torch.no_grad():
        final_dist, p_gen, cross_attn = model.forward_with_copy(enc_input, enc_mask, dec_input, dec_mask)
        
        print("\n--- Diagnostics ---")
        print(f"p_gen value: {p_gen.item():.4f}")
        
        # Check attention entropy (sharpness)
        # cross_attn: [1, 1, src_len]
        probs = cross_attn[0, 0, :]
        entropy = -(probs * (probs + 1e-12).log()).sum()
        max_entropy = np.log(len(src_tokens) + 2)
        print(f"Last-layer Attention Entropy: {entropy:.4f} (Max possible: {max_entropy:.4f})")
        
        # Distribution check
        top_probs, top_ids = torch.topk(final_dist[0, 0, :], 5)
        print(f"Top 5 predicted tokens:")
        for p, tid in zip(top_probs, top_ids):
            print(f"  {tokenizer.decode([tid.item()]):<15} : {p.item():.4f} (id: {tid.item()})")

if __name__ == "__main__":
    health_check()
