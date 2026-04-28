import torch
from tokenizer_utils import get_tokenizer
from pretrain_config import get_finetune_config
from model import build_transformer
from checkpoint_utils import load_checkpoint
from inference import beam_search_decode

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = get_finetune_config()
    tokenizer = get_tokenizer(config['tokenizer_model'])
    
    model = build_transformer(
        src_vocab_size=tokenizer.get_vocab_size(),
        tgt_vocab_size=tokenizer.get_vocab_size(),
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
    
    # LOAD THE PRETRAIN WEIGHTS DIRECTLY
    ckpt_path = 'pretrain_weights_multi_fixed/pretrain_multi_fixed_best.pt'
    checkpoint = load_checkpoint(ckpt_path, map_location=device)
    
    # Handle tgt_pos length mismatch (128 vs 100)
    state_dict = checkpoint['model_state_dict']
    if 'tgt_pos.pe' in state_dict:
        del state_dict['tgt_pos.pe']
    if 'src_pos.pe' in state_dict:
        del state_dict['src_pos.pe']
        
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    text = "The quick brown fox jumps over the lazy dog."
    src_tokens = tokenizer.encode(text)
    enc_input = torch.tensor([src_tokens], dtype=torch.long).to(device)
    enc_mask = torch.ones(1, 1, 1, len(src_tokens)).to(device)
    
    with torch.no_grad():
        enc_output = model.encode(enc_input, enc_mask)
        
        output_tokens = beam_search_decode(
            model, enc_output, enc_mask, enc_input, tokenizer, 
            config['tgt_seq_len'], device, 
            beam_size=4, length_penalty=0.8, no_repeat_ngram=3
        )
        
    print(f"Input: {text}")
    print(f"Output: {tokenizer.decode(output_tokens)}")

if __name__ == '__main__':
    main()
