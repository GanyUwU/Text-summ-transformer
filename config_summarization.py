from pathlib import Path

def get_config():
    """
    FIXED configuration for better convergence.
    
    KEY CHANGES:
    1. Higher learning rate with warmup
    2. Gradient accumulation for effective batch size of 32
    3. More training data (50K samples)
    """
    return {
        # Training hyperparameters
        "batch_size": 8,  # Keep small for memory, but use gradient accumulation
        "gradient_accumulation_steps": 4,  # Effective batch size = 8 * 4 = 32
        "num_epochs": 20,
        "lr": 1e-3,  # Higher LR with warmup
        "warmup_steps": 2000,  # Warmup for first 2000 steps
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        
        # Sequence lengths
        "src_seq_len": 512,
        "tgt_seq_len": 80,
        
        # Model architecture
        "d_model": 256,
        "d_ff": 1024,
        "num_layers": 4,
        "num_heads": 8,
        "dropout": 0.1,  # Reduced from 0.2 for faster learning
        
        # Dataset - USE MORE DATA!
        "datasource": 'cnn_dailymail',
        "dataset_version": '3.0.0',
        "train_samples": 50000,  # Increased from 10K
        "val_samples": 2000,
        "lang": "en",
        
        # Paths
        "model_folder": "weights_v2",
        "model_basename": "summarizer_",
        "preload": None,
        "tokenizer_file": "tokenizer_summarization_{0}.json",
        "experiment_name": "runs/summarization_v2",
        
        # Loss
        "label_smoothing": 0.1,
        
        # Validation
        "num_validation_examples": 3,
        
        # Early stopping
        "patience": 5,
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])