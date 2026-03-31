from pathlib import Path
import torch
import os

def load_checkpoint(path, map_location='cpu'):
    """Robustly load a checkpoint.

    Tries torch.load first. On UnpicklingError or known loader failures,
    will attempt to load via safetensors (if installed) or retry with
    latin1 encoding on failure. Raises a clear RuntimeError if all attempts fail.
    """
    path = str(path)
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except Exception as e:
        # Best-effort fallbacks
        msg = str(e).lower()
        # Try safetensors if file looks like that or error mentions safetensors
        if path.endswith('.safetensors') or 'safetensors' in msg:
            try:
                from safetensors.torch import load_file as _safetensors_load
                print(f"Attempting safetensors.torch.load_file for {path}")
                return {'model_state_dict': _safetensors_load(path, device=map_location)}
            except Exception:
                pass

        # Retry with latin1 encoding (useful for Python2->3 pickles)
        try:
            print(f"Retrying torch.load for {path} with encoding='latin1'...")
            return torch.load(path, map_location=map_location, encoding='latin1', weights_only=False)
        except Exception:
            pass

        raise RuntimeError(
            f"Checkpoint load failed for {path}.\n"
            "If this is a safetensors file, install 'safetensors' (pip install safetensors) and try again.\n"
            "If the file is from a different framework or corrupted, verify its origin.\n"
            f"Original error: {repr(e)}"
        ) from e
