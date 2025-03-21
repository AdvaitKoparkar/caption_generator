"""
Script for fine-tuning the Microsoft GIT (Generative Image-to-Text) model on image captioning tasks.
This module provides functionality to train the model on custom datasets with configurable parameters
and logging capabilities using Weights & Biases.
"""

import os
import torch
import wandb
from tqdm import tqdm
from typing import Dict, Any, Tuple, Optional
import datetime

from transformers import AutoProcessor
from transformers import AutoModelForCausalLM
from datasets import load_dataset

class GiTFineTuningConfig:
    """
    Configuration class for GIT model fine-tuning parameters.
    Contains all hyperparameters and training settings.
    """
    model_name = "microsoft/git-base"  # Base model to fine-tune
    seed = 2025  # Random seed for reproducibility
    num_samples = 128  # Number of samples to use for training
    train_split = 0.8  # Proportion of data to use for training
    validate_every = 15  # Number of epochs between validation
    batch_size = 32  # Batch size for training
    num_epochs = 2  # Total number of training epochs
    optimizer_config = {
        'lr': 1e-5,  # Learning rate for Adam optimizer
    }
    checkpoint_dir = "checkpoints"  # Base directory for checkpoints

    def get_run_checkpoint_dir(run_name: str) -> str:
        """
        Get the checkpoint directory for a specific run.
        
        Args:
            run_name: Name of the training run
            
        Returns:
            Path to the run's checkpoint directory
        """
        return os.path.join(GiTFineTuningConfig.checkpoint_dir, run_name)

class ImageCaptioningDataset(torch.utils.data.Dataset):
    """
    Custom dataset class for image captioning tasks.
    Handles preprocessing of images and captions using the GIT processor.
    
    Args:
        data: Dataset containing image-caption pairs
        processor: GIT processor for tokenizing and processing inputs
    """
    def __init__(self, data: torch.utils.data.Dataset, processor: AutoProcessor):
        self.data = data
        self.processor = processor

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns processed image-caption pair at the given index.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Dictionary containing processed inputs ready for model
        """
        item = self.data[idx]
        encoding = self.processor(
            images=item["image"], 
            text=item["caption"], 
            padding="max_length", 
            return_tensors="pt",
            return_attention_mask=True  # Explicitly request attention mask
        )
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        return encoding

class Trainer:
    """
    Trainer class for fine-tuning the GIT model.
    Handles training loop, validation, and logging to Weights & Biases.
    
    Args:
        run: Name of the training run
        model: GIT model to train
        dataloader_train: DataLoader for training data
        dataloader_val: DataLoader for validation data
        optimizer: Optimizer for model training
    """
    def __init__(self, run: str,
                 model: torch.nn.Module, 
                 dataloader_train: torch.utils.data.DataLoader, 
                 dataloader_val: torch.utils.data.DataLoader, 
                 optimizer: torch.optim.Optimizer):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.optimizer = optimizer
        self.epochs_run = 0
        self.best_val_loss = float('inf')
        self.model = model.to(self.device)
        self.run_name = run
        
        # Create checkpoint directory for this run
        self.checkpoint_dir = GiTFineTuningConfig.get_run_checkpoint_dir(run)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Find and load best checkpoint if exists
        best_checkpoint = self._find_best_checkpoint()
        if best_checkpoint:
            print(f"Found best checkpoint for run '{run}': {best_checkpoint}")
            self._load_checkpoint(best_checkpoint)
        
        # Initialize Weights & Biases logging
        self.logger = wandb.init(
            project="caption-generator-GiT",
            name=f"experiment_{run}",
            config={
                'model_name': GiTFineTuningConfig.model_name,
                'seed': GiTFineTuningConfig.seed,
                'num_samples': GiTFineTuningConfig.num_samples,
                'train_split': GiTFineTuningConfig.train_split,
                'validate_every': GiTFineTuningConfig.validate_every,
                'batch_size': GiTFineTuningConfig.batch_size,
                'num_epochs': GiTFineTuningConfig.num_epochs,
                'optimizer_config': GiTFineTuningConfig.optimizer_config,
            }
        )

    def _find_best_checkpoint(self) -> Optional[str]:
        """
        Finds the best checkpoint for the current run based on validation loss.
        
        Returns:
            Path to the best checkpoint if found, None otherwise
        """
        if not os.path.exists(self.checkpoint_dir):
            return None
            
        # Find all checkpoints for this run
        checkpoints = []
        for file in os.listdir(self.checkpoint_dir):
            if file.endswith('.pt'):
                checkpoint_path = os.path.join(self.checkpoint_dir, file)
                try:
                    # Load checkpoint to get validation loss
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    checkpoints.append((checkpoint_path, checkpoint['val_loss']))
                except Exception as e:
                    print(f"Warning: Could not load checkpoint {file}: {e}")
                    continue
        
        if not checkpoints:
            return None
            
        # Return checkpoint with lowest validation loss
        return min(checkpoints, key=lambda x: x[1])[0]

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Loads a checkpoint and restores model and optimizer state.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore training state
        self.epochs_run = checkpoint['epochs_run']
        self.best_val_loss = checkpoint['best_val_loss']
        
        # Log checkpoint loading information
        self.logger.log({
            "checkpoint_loaded": True,
            "checkpoint_path": checkpoint_path,
            "checkpoint_epoch": checkpoint['epoch'],
            "checkpoint_val_loss": checkpoint['val_loss'],
            "checkpoint_best_val_loss": checkpoint['best_val_loss'],
            "previous_wandb_run_id": checkpoint['wandb_run_id']
        })

    def _step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Performs a single training step.
        
        Args:
            batch: Dictionary containing input tensors
            
        Returns:
            Loss value for the current step
        """
        self.optimizer.zero_grad()
        input_ids = batch.pop("input_ids").to(self.device)
        pixel_values = batch.pop("pixel_values").to(self.device)
        attention_mask = batch.pop("attention_mask").to(self.device)
        
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            labels=input_ids
        )
        loss = outputs.loss
        loss.backward()
        self.optimizer.step()
        
        # Log per-batch metrics
        self.logger.log({
            "batch_loss": loss.item(),
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "batch_size": input_ids.size(0),
            "avg_attention_mask": attention_mask.float().mean().item()  # Log average mask value
        })
        return loss.item()

    def _run_epoch(self, epoch: int) -> float:
        """
        Runs a single training epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average loss for the epoch
        """
        pbar = tqdm(self.dataloader_train)
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in pbar:
            loss = self._step(batch)
            epoch_loss += loss
            num_batches += 1
            pbar.set_description(f"Epoch {epoch} Loss: {loss:.4f}")
            
        avg_epoch_loss = epoch_loss / num_batches
        self.logger.log({
            "epoch": epoch,
            "epoch_loss": avg_epoch_loss,
            "num_batches": num_batches
        })
        return avg_epoch_loss
        
    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """
        Saves a checkpoint of the model, optimizer, and training state.
        
        Args:
            epoch: Current epoch number
            val_loss: Current validation loss
        """
        # Create a timestamp for unique checkpoint naming
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"epoch{epoch}_val{val_loss:.4f}_{timestamp}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'epochs_run': self.epochs_run,
            'wandb_run_id': self.logger.id,
            'run_name': self.run_name,
            'timestamp': timestamp
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.log({
            "checkpoint_saved": True,
            "checkpoint_path": checkpoint_path,
            "checkpoint_name": checkpoint_name,
            "checkpoint_epoch": epoch,
            "checkpoint_val_loss": val_loss,
            "checkpoint_timestamp": timestamp
        })

    def _validate(self) -> float:
        """
        Runs validation on the validation dataset.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.dataloader_val, desc="Validating"):
                input_ids = batch.pop("input_ids").to(self.device)
                pixel_values = batch.pop("pixel_values").to(self.device)
                attention_mask = batch.pop("attention_mask").to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                loss = outputs.loss.item()
                total_val_loss += loss
                num_batches += 1
                
                # Log sample predictions every few batches
                if num_batches % 5 == 0:  # Log every 5th batch
                    # Generate captions for a sample
                    generated_ids = self.model.generate(
                        pixel_values=pixel_values[:1],  # Take first image in batch
                        attention_mask=attention_mask[:1],  # Use attention mask for generation
                        max_length=50,
                        num_beams=5,
                        early_stopping=True
                    )
                    generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    true_caption = self.processor.batch_decode(input_ids[:1], skip_special_tokens=True)[0]
                    
                    self.logger.log({
                        "sample_prediction": {
                            "true_caption": true_caption,
                            "generated_caption": generated_caption,
                            "batch": num_batches
                        }
                    })
        
        avg_val_loss = total_val_loss / num_batches
        self.logger.log({
            "validation_loss": avg_val_loss,
            "validation_batches": num_batches
        })
        return avg_val_loss
        
    def train(self) -> None:
        """
        Runs the complete training process for the configured number of epochs.
        Includes validation steps based on validate_every parameter.
        """
        try:
            for epoch in range(self.epochs_run, GiTFineTuningConfig.num_epochs):
                # Training step
                avg_loss = self._run_epoch(epoch)
                
                # Validation step if needed
                if (epoch + 1) % GiTFineTuningConfig.validate_every == 0:
                    val_loss = self._validate()
                    
                    # Update best validation loss and save checkpoint
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint(epoch, val_loss)
                        self.logger.log({
                            "best_val_loss": self.best_val_loss,
                            "epoch_of_best_val": epoch
                        })
                
                self.epochs_run += 1  # Update the number of epochs run
        finally:
            self.logger.finish()  # Ensure logger is closed even if training is interrupted

def load_training_artifact() -> Tuple[torch.nn.Module, torch.utils.data.Dataset, torch.utils.data.Dataset, torch.optim.Optimizer]:
    """
    Loads and prepares all necessary components for training.
    
    Returns:
        Tuple containing (model, training dataset, validation dataset, optimizer)
    """
    # Load dataset and processor
    data = load_dataset("mrSoul7766/instagram_post_captions", split=f'train[0:{GiTFineTuningConfig.num_samples}]')
    processor = AutoProcessor.from_pretrained(GiTFineTuningConfig.model_name)
    
    # Create and split dataset
    dataset = ImageCaptioningDataset(data, processor)
    data_train, data_val = torch.utils.data.random_split(
        dataset, 
        [GiTFineTuningConfig.train_split, 1-GiTFineTuningConfig.train_split]
    )
    
    # Initialize model and optimizer
    model = AutoModelForCausalLM.from_pretrained(GiTFineTuningConfig.model_name)
    opt = torch.optim.Adam(model.parameters(), **GiTFineTuningConfig.optimizer_config)
    return model, data_train, data_val, opt

def get_dataloader(dataset: torch.utils.data.Dataset) -> torch.utils.data.DataLoader:
    """
    Creates a DataLoader for the given dataset with configured batch size.
    
    Args:
        dataset: Dataset to create DataLoader for
        
    Returns:
        Configured DataLoader instance
    """
    dl = torch.utils.data.DataLoader(
        dataset, 
        batch_size=GiTFineTuningConfig.batch_size,
        pin_memory=True,
        shuffle=True,
    )
    return dl

def train(run_name: str = 'test') -> None:
    """
    Main training function that orchestrates the entire training process.
    
    Args:
        run_name: Name of the training run
    """
    # Load all necessary components
    model, data_train, data_val, opt = load_training_artifact()
    
    # Create data loaders
    dataloader_train = get_dataloader(data_train)
    dataloader_val = get_dataloader(data_val)
    
    # Initialize trainer and start training
    trainer = Trainer(run_name, model, dataloader_train, dataloader_val, opt)
    trainer.train()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train the GIT model for image captioning')
    parser.add_argument('--run_name', type=str, default='test', help='Name of the training run')
    
    args = parser.parse_args()
    
    wandb.login()
    train(args.run_name)