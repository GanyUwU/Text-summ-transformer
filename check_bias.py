import torch
from pathlib import Path

def main():
    ckpt_path = 'weights_v11_nuclear/nuclear_summarizer_best.pt'
    if not Path(ckpt_path).exists():
        print("Checkpoint not found!")
        return
        
    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in ckpt:
        d = ckpt['model_state_dict']
    else:
        d = ckpt
        
    bias = d.get('copy_mechanism.w_gen.bias')
    weight = d.get('copy_mechanism.w_gen.weight')
    
    if bias is not None:
        print(f"w_gen.bias = {bias.item():.4f}")
    else:
        print("w_gen.bias is MISSING in checkpoint!")
        
    if weight is not None:
        print(f"w_gen.weight shape = {weight.shape}, mean = {weight.mean().item():.4f}, std = {weight.std().item():.4f}")
    else:
        print("w_gen.weight is MISSING!")

if __name__ == '__main__':
    main()
