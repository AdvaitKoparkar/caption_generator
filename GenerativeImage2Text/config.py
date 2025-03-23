"""
Configuration module for the Generative Image to Text (GIT) model training.
"""

import os
import json
from typing import Optional

class GiTFineTuningConfig:
    """
    Configuration class for GIT model fine-tuning parameters.
    Contains all hyperparameters and training settings.
    """
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from JSON file or use defaults.
        
        Args:
            config_path: Path to JSON configuration file. If None, uses defaults.
        """
        # Default configuration
        self.model_name = "microsoft/git-base"
        self.seed = 2025
        self.num_samples = 128
        self.train_split = 0.8
        self.validate_every = 1
        self.batch_size = 32
        self.num_epochs = 2
        self.optimizer_config = {
            "lr": 1e-5
        }
        self.scheduler_config = {
            "type": "linear",
            "warmup_steps": 100,
            "num_training_steps": None
        }
        self.training_config = {
            "gradient_clip_val": 1.0,
            "accumulate_grad_batches": 1,
            "precision": "32"
        }
        self.early_stopping_config = {
            "patience": 3,
            "min_delta": 0.001,
            "mode": "min"
        }
        self.validation_config = {
            "visualize_every_n_batches": 1
        }
        self.checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
        self.generation_config = {
            "max_length": 50,
            "num_beams": 5,
            "early_stopping": True
        }
        
        # Load configuration from file if provided
        if config_path:
            self.load_from_json(config_path)
    
    def load_from_json(self, config_path: str) -> None:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to JSON configuration file
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
            
        with open(config_path, "r") as f:
            config = json.load(f)
            
        # Update configuration with values from JSON
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown configuration key '{key}' in {config_path}")
    
    def save_to_json(self, config_path: str) -> None:
        """
        Save current configuration to JSON file.
        
        Args:
            config_path: Path to save JSON configuration file
        """
        config_dict = {
            "model_name": self.model_name,
            "seed": self.seed,
            "num_samples": self.num_samples,
            "train_split": self.train_split,
            "validate_every": self.validate_every,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "optimizer_config": self.optimizer_config,
            "scheduler_config": self.scheduler_config,
            "training_config": self.training_config,
            "early_stopping_config": self.early_stopping_config,
            "validation_config": self.validation_config,
            "checkpoint_dir": self.checkpoint_dir,
            "generation_config": self.generation_config
        }
        
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
    
    def get_run_checkpoint_dir(self, run_name: str) -> str:
        """
        Get the checkpoint directory for a specific run.
        
        Args:
            run_name: Name of the training run
            
        Returns:
            Path to the run's checkpoint directory
        """
        return os.path.join(self.checkpoint_dir, run_name) 