import torch
from tokenizer_utils import get_tokenizer
from pretrain_config import get_finetune_config
from model import build_transformer
from checkpoint_utils import load_checkpoint

def main():
    device = torch.device('cpu')
    config = get_finetune_config()
    tokenizer = get_tokenizer(config['tokenizer_model'])
    
    # Load Model
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
    
    checkpoint = load_checkpoint('weights_v11_nuclear/nuclear_summarizer_best.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Dummy input
    text = "The quick brown fox jumps over the lazy dog."
    src_tokens = tokenizer.encode(text)
    src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(device)
    src_mask = torch.ones(1, 1, 1, len(src_tokens)).to(device)
    
    dec_tokens = [tokenizer.bos_id]
    dec_tensor = torch.tensor([dec_tokens], dtype=torch.long).to(device)
    dec_mask = torch.ones(1, 1, 1, 1).to(device)
    
    print(f"w_gen.bias = {model.copy_mechanism.w_gen.bias.item():.4f}")
    
    with torch.no_grad():
        encoder_output = model.encode(src_tensor, src_mask)
        decoder_output, cross_attn = model.decode(
            encoder_output, src_mask, dec_tensor, dec_mask, return_cross_attn=True
        )
        vocab_logits = model.project(decoder_output)
        tgt_embed = model.tgt_pos(model.tgt_embed(dec_tensor))
        context_vector = torch.bmm(cross_attn, encoder_output)
        final_dist, p_gen = model.copy_mechanism(
            decoder_output, context_vector, tgt_embed, vocab_logits, cross_attn, src_tensor
        )
        
        print(f"p_gen value: {p_gen.mean().item():.4f}")
        print(f"vocab_logits max: {vocab_logits.max().item():.4f}")
        print(f"cross_attn max: {cross_attn.max().item():.4f}")

if __name__ == '__main__':
    main()
