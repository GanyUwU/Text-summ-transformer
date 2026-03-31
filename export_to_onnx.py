import torch
import onnx
import argparse
from pathlib import Path
import sys

# Add root path
sys.path.append(str(Path(__file__).parent))

from model import build_transformer
from pretrain_config import get_finetune_config
from checkpoint_utils import load_checkpoint

def export_model_to_onnx(checkpoint_path, output_path="pretrain_model_debug.onnx", config=None):
    if config is None:
        config = get_finetune_config()
    
    device = torch.device("cpu") # Exporting on CPU is safer for ONNX compatibility
    
    # Configuration matches our Phase 10 architecture
    d_model = config['d_model']
    d_ff = config['d_ff']
    num_layers = config['num_layers']
    num_heads = config['num_heads']
    dropout = 0.0 # No dropout for inference/debug
    src_vocab_size = config['vocab_size']
    tgt_vocab_size = config['vocab_size']
    src_seq_len = config['src_seq_len']
    tgt_seq_len = config['tgt_seq_len']
    
    print(f"Building model with d_model={d_model}, layers={num_layers}...")
    model = build_transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_seq_len=src_seq_len,
        tgt_seq_len=tgt_seq_len,
        d_model=d_model,
        N=num_layers,
        h=num_heads,
        dropout=dropout,
        d_ff=d_ff,
        use_copy=True # Always enable for the full suite
    )
    
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Dummy inputs for ONNX tracer
    # [batch, seq_len]
    dummy_encoder_input = torch.randint(0, src_vocab_size, (1, src_seq_len), dtype=torch.long)
    dummy_decoder_input = torch.randint(0, tgt_vocab_size, (1, tgt_seq_len), dtype=torch.long)
    
    # Masks
    dummy_encoder_mask = torch.ones((1, 1, 1, src_seq_len), dtype=torch.bool)
    dummy_decoder_mask = torch.ones((1, 1, tgt_seq_len, tgt_seq_len), dtype=torch.bool)
    
    print(f"Exporting to {output_path}...")
    # We export the full forward pass
    torch.onnx.export(
        model,
        (dummy_encoder_input, dummy_encoder_mask, dummy_decoder_input, dummy_decoder_mask),
        output_path,
        export_params=True,
        opset_version=17, # Modern version with Softmax axis support
        do_constant_folding=True,
        input_names=['encoder_input', 'encoder_mask', 'decoder_input', 'decoder_mask'],
        output_names=['logits'],
        
        dynamic_axes={
            'encoder_input': {0: 'batch', 1: 'src_seq'},
            'encoder_mask': {0: 'batch', 3: 'src_seq'},
            'decoder_input': {0: 'batch', 1: 'tgt_seq'},
            'decoder_mask': {0: 'batch', 2: 'tgt_seq', 3: 'tgt_seq'}
        }
    )
    
    print("Export Complete! Model ready for Transformer Debug Suite.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", type=str, default="model_debug.onnx", help="Output .onnx filename")
    args = parser.parse_args()
    
    export_model_to_onnx(args.checkpoint, args.output)
