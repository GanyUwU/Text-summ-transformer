"""
Professional Pretraining Configuration

Optimized for 8GB GPU with Wikipedia pretraining.
"""

# from pathlib import Path


# def get_pretrain_config():
#     """
#     Configuration for pretraining on Wikipedia with denoising objective.
    
#     Optimized for:
#     - 8GB VRAM GPU
#     - ~10k steps (few hours training)
#     - Full Wikipedia dataset
#     """
#     return {
#         # Model Architecture (medium size for 8GB GPU)
#         "d_model": 512,
#         "d_ff": 2048,           # 4x d_model
#         "num_layers": 6,        # 6 encoder + 6 decoder
#         "num_heads": 8,
#         "dropout": 0.2,  # INCREASED from 0.1: stronger regularization to fight overfitting
#         "share_weights": True,   # Share src/tgt/projection embeddings
#         "use_copy": False,       # No copy for pretraining
        
#         # Tokenizer
#         "tokenizer_model": "tokenizer_sp.model",
#         "vocab_size": 32000,
        
#         # Sequence length (shorter for pretraining)
#         "seq_len": 256,
        
#         # Denoising parameters (BART-style) - FIXED masking
#         "mask_prob": 0.30,          # 30% of tokens masked (was broken: applying 2x probability)
#         "mask_span_lambda": 3.0,    # Average span length
#         "shuffle_sentences": True,  # Mix up sentence order for harder task
        
#         # Training
#         "batch_size": 4,            # Small batch for 8GB GPU
#         "gradient_accumulation": 8,  # Effective batch = 32
#         "num_steps": 25000,         # INCREASED: Need ~25k steps for 500k articles
#         "lr": 1e-3,                 # FIXED: Normal LR for learning
#         "warmup_steps": 500,        # 
#         "weight_decay": 0.05,       # INCREASED from 0.01: stronger L2 regularization for overfitting
#         "grad_clip": 1.0,
#         "label_smoothing": 0.0,
        
#         # Diagnostics (Active fixes)
#         "entropy_reg_weight": 0.0,       # Disabled: let model learn naturally without forcing divergence
#         "decoder_lr_scale": 1.0,          # FIXED: Normal LR for decoder (was 0.33 - 3x too slow!)
#         "reinit_decoder_heads": False,    # Not needed when starting fresh
#         "resume_from": None,              # FIXED: Start fresh (was loading collapsed checkpoint!)
#         "save_every": 500,
        
#         # Mixed precision (AMP)
#         "use_amp": True,
        
#         # Gradient checkpointing (saves memory)
#         "gradient_checkpointing": True,
        
#         # Data
#         "dataset": "wikipedia",
#         "dataset_config": "20231101.en",
#         "max_articles": 500000,  # INCREASED 10x: ~1 billion tokens (was 100M, causing overfitting)
        
#         # Checkpointing
#         "save_every": 500,
#         "model_folder": "pretrain_weights_fixed",
#         "model_basename": "pretrain_fixed_",
#         "experiment_name": "runs/pretrain_fixed",
        
#         # Logging
#         "log_every": 100,
#     }


def get_finetune_config():
    """
    Configuration for training on CNN/DailyMail.
    
    Optimized for ~60M param model with weight sharing + copy mechanism.
    """
    return {
        # Model architecture MUST MATCH pretrained checkpoint
        "d_model": 512,
        "d_ff": 2048,
        "num_layers": 6,
        "num_heads": 8,
        "dropout": 0.2,           # INCREASED from 0.1 for stronger regularization
        "share_weights": True,   # Share src/tgt/projection embeddings
        "use_copy": True,        # Pointer-generator copy mechanism ON for fine-tuning
        
        # Tokenizer
        "tokenizer_model": "tokenizer_sp.model",
        "vocab_size": 32000,
        
        # Sequence lengths
        "src_seq_len": 512,
        "tgt_seq_len": 100,   # Reduced for XSum/SAMSum conciseness
        
        # Training (from-scratch: higher LR + more epochs)
        # Training
        "batch_size": 4,
        "gradient_accumulation": 16, # INCREASED (was 8) for effective batch = 64
        "num_epochs": 4,            # Phase 11: Nuclear Option (Long-lead restart)
        "lr": 1.5e-5,                  # Phase 11: Stable fine-tuning rate
        "warmup_steps": 1000,        # Standard warmup for fresh restart from pretrain
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "label_smoothing": 0.1,      # REVERTED to 0.1 for stability
        
        # Mixed precision
        "use_amp": True,
        
        # Data (Optimized for 1-hour feedback loop)
        "train_samples": 50000,      # Faster iterations (was 200k)
        "val_samples": 200,
        
        "datasource": "cnn_dailymail",
        "dataset_version": "3.0.0",
        
        # Phase 11: Nuclear Option (Start from RAW Uncorrupted Weights)
        "pretrain_weights": "pretrain_weights_multi_fixed/pretrain_multi_fixed_best.pt",
        "preload": None,
        # Inside get_finetune_config()
        #"pretrain_weights": "/kaggle/input/datasets/ppbbb1123/trail-transformer/nuclear_summarizer_epoch_4.pt",
        #"tokenizer_model": "/kaggle/input/datasets/ppbbb1123/trail-transformer/tokenizer_sp.model",

        
        # Evaluation & Diagnostics
        "save_every": 500,           # High-frequency validation (~5 times per epoch)
        "diagnostic_every": 50,      # Monitor p_gen behavior early
        "patience": 10,
        "eval_metric": "rouge1",
        
        # Phase 17.2: The Nuclear Detox (Initial Phase)
        # Goal: Violently break the identity map addiction.
        "coverage_loss_weight": 0.6,    # High factual anchoring
        "entropy_reg_weight": 0.002,   # High diversity pressure (Initial Detox)
        "target_entropy": 1.6,         
        "lead_mask_prob": 0.9,         # High feature starvation (Force unlearning)
        "decoder_lr_scale": 2.0,      # Stabilized from 10.0
        
        # Checkpointing
        "model_folder": "weights_v11_nuclear",
        "model_basename": "nuclear_summarizer_",
        "experiment_name": "runs/train_v11_nuclear_final",
        
        
        "beam_size": 4,
        "length_penalty": 1.0,       # Concise (Phase 10)
        "no_repeat_ngram": 3,
        "num_validation_examples": 200, # Matches val_samples
        # AGGRESSIVE NUCLEAR FIXES per Lin et al. / Boutkan et al. (Breaking copying plateau)
        "pgen_loss_weight": 1.0,        # Legacy (unused with lambda_p)
        "pgen_target": 0.45,
        # AGGRESSIVE LINEAR PENALTY for copying (replaces MSE-based pgen_balance_loss)
        "lambda_p": 3.0,  # HIGH multiplier on mean(1 - p_gen); increase if copy rate stays high
        # HARD POINTER DROPOUT: force p_gen = 1.0 for ~20% of batches (breaks copying shortcut)
        "hard_pointer_dropout_prob": 0.3, # Increased from 0.2 to safely break identity map
        # GENERATOR WARMUP: disable copy mechanism for first N steps
        "copy_warmup_steps": 1200,  # Increased for surgical detox stability
        # Pointer-dropout (legacy; superseded by hard_pointer_dropout_prob)
        "pointer_dropout": 0.0,
        # Scheduling
        "apply_pointer_loss_after_steps": 0,
        "apply_coverage_after_steps": 0,
        # Cockpit for fine-tuning monitoring
        "enable_cockpit": False,
        "cockpit_out": "diagnostics_output/cockpit",
        "reinit_decoder_heads": False,  # Preserve specialized heads from Step 25k
        # Dataset mix disabled, using pure CNN/DailyMail
        "dataset_mix": None,
        # Staged unfreeze: keep encoder effectively frozen and ramp its LR over this many steps
        "freeze_steps": 400,  # ~0.5 Epoch for 50k samples (was 1000)
        "staged_unfreeze": True,
    }   
#         # Diagnostics (from reviewer's checklist)
#         "entropy_reg_weight": 1e-3,       # Attention entropy regularization
#         "coverage_loss_weight": 0.1,      # Coverage loss for cross-attention
#         "decoder_lr_scale": 0.33,         # Decoder LR = base_lr * this
#         "reinit_decoder_heads": False,    # Preserve pre-trained weights in Phase 3
#         "diagnostic_every": 100,          # Log diagnostics every N steps
#         "save_every": 100,                # Temporary for debugging validation health
#     }


# def get_pretrain_weights_path(config, identifier: str):
#     """Get path for saving pretrain weights."""
#     folder = Path(config['model_folder'])
#     folder.mkdir(parents=True, exist_ok=True)
#     return str(folder / f"{config['model_basename']}{identifier}.pt")


# def get_latest_pretrain_weights(config):
#     """Get latest pretrain checkpoint."""
#     folder = Path(config['model_folder'])
#     if not folder.exists():
#         return None
    
#     pattern = f"{config['model_basename']}*.pt"
#     files = list(folder.glob(pattern))
    
#     if not files:
#         return None
    
#     # Sort by modification time
#     files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
#     return str(files[0])


# # For backward compatibility
# def get_config():
#     """Alias for get_finetune_config."""
#     return get_finetune_config()


def get_multi_dataset_config():
    config = get_pretrain_config()
    config.update({
        # Training — STABLE from-scratch pretraining
        "lr": 1e-4,                 # ↓ from 1e-3 (main fix)
        "warmup_steps": 2000,      # ↑ from 1000 (more stable start)
        "num_steps": 400000,        # 400k micro ÷ 16 accum = 25k optimizer steps
        "dropout": 0.2,
        "weight_decay": 0.01,      # ↓ from 0.
        "grad_clip": 0.5,

        # Datasets — keep as-is
        "datasets": ["wikipedia", "bookcorpus", "openwebtext"],
        "max_articles": 1500000,

        # Diagnostics — monitor, don’t fight numerics
        "entropy_reg_weight": 1e-3,#5e-3,    # ↓ from 1e-3 (or set to 0.0 initially)
        "target_entropy": 2.0,         # ↓ from 2.5 (more realistic for deep layers)
        "decoder_lr_scale": 1.0,
        "reinit_decoder_heads": False,   # don’t reinit during a run
        "reinit_only_decoder": True,     # if you ever reinit, decoder only
        "diversity_weight": 0.0,#1e-2,
        "label_smoothing": 0.1,         # fights output overconfidence

        # Safety — don’t resume broken runs
        "resume_from": "pretrain_weights_multi_fixed/pretrain_multi_fixed_step_18500.pt",

        #"debug_samples": 20000,

        # Cockpit
        "enable_cockpit": True,
        "cockpit_out": "diagnostics_output/cockpit",

        # Checkpointing
        "model_folder": "pretrain_weights_multi_fixed",
        "model_basename": "pretrain_multi_fixed_",
        "experiment_name": "runs/pretrain_multi_stable_final",
    })
    return config




from pathlib import Path

def get_pretrain_config():

    return {
        # Model Architecture
        "d_model": 512,
        "d_ff": 2048,
        "num_layers": 6,
        "num_heads": 8,
        "dropout": 0.2,
        "share_weights": True,
        "use_copy": False,

        # Tokenizer
        "tokenizer_model": "tokenizer_sp.model",
        "vocab_size": 32000,

        # Sequence length
        "seq_len": 256,
        # Curriculum for sequence length during pretraining: start small and increase
        # to teach long-range dependencies without huge memory spikes.
        # Example progression: 256 -> 512 -> 1024 -> 2000
        "curriculum_seq_lens": [256, 512, 1024, 2000],

        # Denoising params
        "mask_prob": 0.30,
        "mask_span_lambda": 3.0,
        "shuffle_sentences": True,

        # Training
        "batch_size": 4,
        "gradient_accumulation": 16,
        "num_steps": 25000,
        "lr": 3e-4,                # SAFER base LR for from-scratch
        "warmup_steps": 1000,     # Longer warmup to avoid early collapse
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "label_smoothing": 0.0,

        # Diagnostics & regularization
        "entropy_reg_weight": 1e-3,
        "decoder_lr_scale": 0.33,
        "reinit_decoder_heads": False,
        "reinit_only_decoder": True,
        "resume_from": None,
        "save_every": 500,

        # Mixed precision
        "use_amp": True,
        "gradient_checkpointing": True,

        # Data
        "dataset": "wikipedia",
        "dataset_config": "20231101.en",
        "max_articles": 500000,  # increase for better generalization

        # Checkpoints & logging
        "model_folder": "pretrain_weights_fixed",
        "model_basename": "pretrain_fixed_",
        "experiment_name": "runs/pretrain_fixed",
        "log_every": 100,
    }


# Removed duplicate get_multi_dataset_config definition.



def get_pretrain_weights_path(config, identifier: str):
    folder = Path(config['model_folder'])
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder / f"{config['model_basename']}{identifier}.pt")


def get_latest_pretrain_weights(config):
    folder = Path(config['model_folder'])
    if not folder.exists():
        return None
    pattern = f"{config['model_basename']}*.pt"
    files = list(folder.glob(pattern))
    if not files:
        return None
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(files[0])



if __name__ == '__main__':
    print("Pretrain Config:")
    for k, v in get_pretrain_config().items():
        print(f"  {k}: {v}")
    
    print("\nFinetune Config:")
    for k, v in get_finetune_config().items():
        print(f"  {k}: {v}")

#         print(f"  {k}: {v}")
