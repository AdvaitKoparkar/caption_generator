"""
Model wrapper module for the Generative Image to Text (GIT) model.
Provides a simple API for loading and using fine-tuned models.
"""

import os
import torch
from typing import Optional, List, Union
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM

from .config import GiTFineTuningConfig

class GiT:
    """
    Wrapper class for the GIT model that provides a simple API for loading and using fine-tuned models.
    Falls back to the base model if no checkpoints are available.
    
    Args:
        checkpoint_dir: Directory containing model checkpoints
        run_name: Name of the training run to load
        device: Device to load the model on ("cuda" or "cpu")
    """
    def __init__(self, 
                 checkpoint_dir: str,
                 run_name: str,
                 device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = checkpoint_dir
        self.run_name = run_name
        self.is_fine_tuned = False
        
        try:
            # Try to load fine-tuned model
            self._load_fine_tuned_model()
            self.is_fine_tuned = True
        except (FileNotFoundError, Exception) as e:
            print("No checkpoints found, using base Git model")
            self._load_base_model()
    
    def _load_fine_tuned_model(self):
        """Load the fine-tuned model from checkpoints."""
        # Load configuration
        self.config = GiTFineTuningConfig()
        self.config_path = os.path.join(self.checkpoint_dir, self.run_name, "config.json")
        if os.path.exists(self.config_path):
            self.config.load_from_json(self.config_path)
        
        # Initialize processor and model
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        
        # Load best checkpoint
        self._load_best_checkpoint()
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def _load_base_model(self):
        """Load the base Git model."""
        self.processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco").to(self.device)
        self.model.eval()
    
    def _find_best_checkpoint(self) -> Optional[str]:
        """
        Finds the best checkpoint for the current run based on validation loss.
        
        Returns:
            Path to the best checkpoint if found, None otherwise
        """
        run_dir = os.path.join(self.checkpoint_dir, self.run_name)
        if not os.path.exists(run_dir):
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
            
        # Find all checkpoints for this run
        checkpoints = []
        for file in os.listdir(run_dir):
            if file.endswith(".pt"):
                checkpoint_path = os.path.join(run_dir, file)
                try:
                    # Load checkpoint to get validation loss
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    checkpoints.append((checkpoint_path, checkpoint["val_loss"]))
                except Exception as e:
                    print(f"Warning: Could not load checkpoint {file}: {e}")
                    continue
        
        if not checkpoints:
            raise FileNotFoundError(f"No valid checkpoints found in {run_dir}")
            
        # Return checkpoint with lowest validation loss
        return min(checkpoints, key=lambda x: x[1])[0]
    
    def _load_best_checkpoint(self) -> None:
        """
        Loads the best checkpoint for the current run.
        """
        checkpoint_path = self._find_best_checkpoint()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Log checkpoint information
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Validation loss: {checkpoint['val_loss']:.4f}")
        print(f"Epoch: {checkpoint['epoch']}")
    
    def describe_image(self, 
                      image: Union[Image.Image, np.ndarray],
                      max_length: Optional[int] = None,
                      num_beams: Optional[int] = None,
                      temperature: float = 1.0,
                      top_p: float = 0.9) -> str:
        """
        Generates a caption for the given image.
        
        Args:
            image: PIL Image or numpy array
            max_length: Maximum length of the generated caption
            num_beams: Number of beams for beam search
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            
        Returns:
            Generated caption
        """
        # Process image
        inputs = self.processor(
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Move inputs to device
        pixel_values = inputs["pixel_values"].to(self.device)
        
        # Get generation parameters
        max_length = max_length or (self.config.generation_config["max_length"] if self.is_fine_tuned else 30)
        num_beams = num_beams or (self.config.generation_config["num_beams"] if self.is_fine_tuned else 5)
        
        # Generate caption
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values=pixel_values,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                top_p=top_p,
                early_stopping=self.config.generation_config["early_stopping"] if self.is_fine_tuned else True
            )
        
        # Decode and return caption
        caption = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return caption
    
    def describe_images(self, 
                       image: Union[Image.Image, np.ndarray],
                       num_captions: int = 3) -> List[str]:
        """
        Generates multiple captions for an image with different parameters.
        
        Args:
            image: PIL Image or numpy array
            num_captions: Number of captions to generate
            
        Returns:
            List of generated captions
        """
        if self.is_fine_tuned:
            # Use different temperatures for variety with fine-tuned model
            temperatures = [0.7, 0.9, 1.0]
            return [
                self.describe_image(
                    image,
                    temperature=temp
                )
                for temp in temperatures[:num_captions]
            ]
        else:
            # Use base model with multiple sequences
            inputs = self.processor(
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            outputs = self.model.generate(
                **inputs,
                max_length=30,
                num_return_sequences=num_captions,
                do_sample=True
            )
            return self.processor.batch_decode(outputs, skip_special_tokens=True)
    
    def __call__(self, 
                 image: Union[Image.Image, np.ndarray, List[Union[Image.Image, np.ndarray]]],
                 **kwargs) -> Union[str, List[str]]:
        """
        Makes the model callable for generating captions.
        
        Args:
            image: Single image or list of images
            **kwargs: Additional arguments for caption generation
            
        Returns:
            Generated caption(s)
        """
        if isinstance(image, list):
            return [self.describe_images(img, **kwargs) for img in image]
        else:
            return self.describe_images(image, **kwargs) 