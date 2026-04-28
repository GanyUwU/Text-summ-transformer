import torch
import os

checkpoint_path = "weights_v11_nuclear/nuclear_summarizer_best.pt"
if os.path.exists(checkpoint_path):
    try:
        # Newer torch defaults to weights_only=True, we need False for complex dicts
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"Global Step: {ckpt.get('global_step', 'Unknown')}")
        print(f"Epoch: {ckpt.get('epoch', 'Unknown')}")
        
        # Check bias value
        if 'model_state_dict' in ckpt:
            sd = ckpt['model_state_dict']
            if 'copy_mechanism.w_gen.bias' in sd:
                bias = sd['copy_mechanism.w_gen.bias'].item()
                import torch.nn.functional as F
                p_gen = torch.sigmoid(torch.tensor(bias)).item()
                print(f"Copy Mechanism w_gen.bias: {bias:.4f} (p_gen approx: {p_gen:.4f})")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
else:
    print(f"Checkpoint not found at {checkpoint_path}")
