import torch
import os
import glob
import numpy as np

class WeightEvolutionAnalyzer:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoints = self._discover_checkpoints()

    def _discover_checkpoints(self):
        """Finds all .pt files in the directory and sorts them by modification time or epoch number."""
        if not os.path.exists(self.checkpoint_dir):
            return []
        
        # Look for checkpoints matching typical patterns
        pts = glob.glob(os.path.join(self.checkpoint_dir, "*.pt"))
        if not pts:
            pts = glob.glob(os.path.join(self.checkpoint_dir, "**", "*.pt"), recursive=True)
            
        # Try to sort numerically if they have numbers in names, else by mtime
        def sort_key(filepath):
            filename = os.path.basename(filepath)
            import re
            numbers = re.findall(r'\d+', filename)
            if numbers:
                return float(numbers[-1])
            return os.path.getmtime(filepath)
            
        return sorted(pts, key=sort_key)

    def analyze(self, max_checkpoints=10):
        """Analyzes weight evolution across discovered checkpoints."""
        if len(self.checkpoints) < 2:
            return {"error": "Need at least 2 checkpoints to analyze evolution."}

        # Subsample if too many
        if len(self.checkpoints) > max_checkpoints:
            indices = np.linspace(0, len(self.checkpoints) - 1, max_checkpoints, dtype=int)
            selected_ckpts = [self.checkpoints[i] for i in indices]
        else:
            selected_ckpts = self.checkpoints

        evolution_data = {}
        prev_state_dict = None
        
        for idx, ckpt_path in enumerate(selected_ckpts):
            try:
                # Load on CPU to avoid VRAM issues
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                
                # Checkpoint might be the state dict directly, or wrapped in a dict
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif isinstance(checkpoint, dict) and 'encoder.layers.0.self_attention_block.q_proj.weight' in checkpoint:
                    state_dict = checkpoint
                else:
                    # Best effort
                    state_dict = checkpoint
                
                step_name = f"Step/Epoch {idx} ({os.path.basename(ckpt_path)[:15]})"
                
                for name, tensor in state_dict.items():
                    if 'weight' not in name:
                        continue
                    
                    if name not in evolution_data:
                        evolution_data[name] = {"mean": [], "std": [], "l2_dist_from_prev": [], "steps": []}
                    
                    np_tensor = tensor.float().numpy()
                    mean_val = float(np.mean(np_tensor))
                    std_val = float(np.std(np_tensor))
                    
                    dist = 0.0
                    if prev_state_dict is not None and name in prev_state_dict:
                        prev_tensor = prev_state_dict[name].float().numpy()
                        dist = float(np.linalg.norm(np_tensor - prev_tensor))
                        
                    evolution_data[name]["mean"].append(mean_val)
                    evolution_data[name]["std"].append(std_val)
                    evolution_data[name]["l2_dist_from_prev"].append(dist)
                    evolution_data[name]["steps"].append(step_name)
                    
                prev_state_dict = state_dict
            except Exception as e:
                print(f"Failed to process checkpoint {ckpt_path}: {e}")
                
        # Group by component for UI
        summary = {
            "num_checkpoints": len(selected_ckpts),
            "encoder_frozen": self._check_frozen(evolution_data, "encoder"),
            "decoder_movement": self._calculate_avg_movement(evolution_data, "decoder"),
            "copy_mech_movement": self._calculate_avg_movement(evolution_data, "copy_mechanism")
        }
        
        return {
            "evolution": evolution_data,
            "summary": summary
        }

    def _check_frozen(self, data, prefix):
        """Checks if a component's weights are hardly moving.
        
        Frozen threshold scales with sqrt(num_params) of the layer.
        For 512×512 weight matrices: threshold ≈ 0.001.
        Fixed 1e-4 was too aggressive and flagged healthy training as frozen.
        """
        frozen_layers = 0
        total_layers = 0
        for name, stats in data.items():
            if name.startswith(prefix) and len(stats["l2_dist_from_prev"]) > 1:
                avg_dist = np.mean(stats["l2_dist_from_prev"][1:])
                # Dynamic threshold: scale with sqrt(num_params)
                # For a [512, 512] matrix: sqrt(262144) ≈ 512, threshold ≈ 0.001
                # This prevents falsely flagging healthy training as frozen.
                num_params = 1
                for d in (stats.get("shape", [512, 512])):
                    num_params *= d
                threshold = 1.0 / np.sqrt(max(num_params, 1)) * 0.5
                total_layers += 1
                if avg_dist < threshold:
                    frozen_layers += 1
        if total_layers == 0: return False
        return frozen_layers > (total_layers * 0.5)  # Frozen if >50% of layers are stuck

    def _calculate_avg_movement(self, data, prefix):
        dists = []
        for name, stats in data.items():
            if name.startswith(prefix) and len(stats["l2_dist_from_prev"]) > 1:
                dists.append(np.mean(stats["l2_dist_from_prev"][1:]))
        if not dists: return 0.0
        return float(np.mean(dists))
