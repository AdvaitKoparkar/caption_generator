"""
Dataset module for the Generative Image to Text (GIT) model training.
"""

import torch
from typing import Dict, Any
from transformers import AutoProcessor

class ImageCaptioningDataset(torch.utils.data.Dataset):
    """
    Dataset class for image captioning using the GIT model.
    Handles loading and preprocessing of images and captions.
    """
    def __init__(self, data, processor: AutoProcessor):
        """
        Initialize the dataset.
        
        Args:
            data: Dataset containing image URLs and captions
            processor: GIT processor for tokenizing and processing inputs
        """
        self.data = data
        self.processor = processor
        
    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary containing processed inputs for the model
        """
        item = self.data[idx]
        
        # Process the image and caption
        inputs = self.processor(
            images=item["image"],
            text=item["caption"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            return_attention_mask=True
        )
        
        # Remove batch dimension and return
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "pixel_values": inputs["pixel_values"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze()
        } 