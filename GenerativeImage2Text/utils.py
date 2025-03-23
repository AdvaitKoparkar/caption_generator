"""
Utility functions for the Generative Image to Text (GIT) model training.
"""

import torch
from typing import Dict, Any, Tuple
from transformers import AutoProcessor, AutoModelForCausalLM
from datasets import load_dataset

from .config import GiTFineTuningConfig
from .dataset import ImageCaptioningDataset

def load_training_artifact(config: GiTFineTuningConfig) -> Tuple[torch.nn.Module, torch.utils.data.Dataset, torch.utils.data.Dataset, torch.optim.Optimizer, AutoProcessor]:
    """
    Loads and prepares all necessary components for training.
    
    Args:
        config: Training configuration object
        
    Returns:
        Tuple containing (model, training dataset, validation dataset, optimizer, processor)
    """
    # Load dataset and processor
    data = load_dataset("mrSoul7766/instagram_post_captions", split=f"train[0:{config.num_samples}]")
    processor = AutoProcessor.from_pretrained(config.model_name)
    
    # Create and split dataset
    dataset = ImageCaptioningDataset(data, processor)
    data_train, data_val = torch.utils.data.random_split(
        dataset, 
        [config.train_split, 1-config.train_split]
    )
    
    # Initialize model and optimizer
    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    opt = torch.optim.Adam(model.parameters(), **config.optimizer_config)
    return model, data_train, data_val, opt, processor

def get_dataloader(dataset: torch.utils.data.Dataset, config: GiTFineTuningConfig) -> torch.utils.data.DataLoader:
    """
    Creates a DataLoader for the given dataset with configured batch size.
    
    Args:
        dataset: Dataset to create DataLoader for
        config: Training configuration object
        
    Returns:
        Configured DataLoader instance
    """
    dl = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config.batch_size,
        pin_memory=True,
        shuffle=True,
    )
    return dl 