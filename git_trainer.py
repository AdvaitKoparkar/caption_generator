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
import json
import numpy as np

from transformers import AutoProcessor
from transformers import AutoModelForCausalLM
from datasets import load_dataset

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
            "type": "linear",  # Options: "linear", "cosine", "constant"
            "warmup_steps": 100,
            "num_training_steps": None  # Will be set during training
        }
        self.training_config = {
            "gradient_clip_val": 1.0,  # Maximum gradient norm
            "accumulate_grad_batches": 1,  # Number of batches to accumulate gradients
            "precision": "32"  # Options: "32", "16", "bf16"
        }
        self.early_stopping_config = {
            "patience": 3,  # Number of epochs to wait before stopping
            "min_delta": 0.001,  # Minimum change in validation loss to be considered an improvement
            "mode": "min"  # Options: "min" for loss, "max" for metrics
        }
        self.validation_config = {
            "visualize_every_n_batches": 1
        }
        self.checkpoint_dir = "checkpoints"
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
        processor: GIT processor for tokenizing and processing inputs
        config: Training configuration object
    """
    def __init__(self, run: str,
                 model: torch.nn.Module, 
                 dataloader_train: torch.utils.data.DataLoader, 
                 dataloader_val: torch.utils.data.DataLoader, 
                 optimizer: torch.optim.Optimizer,
                 processor: AutoProcessor,
                 config: GiTFineTuningConfig):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.optimizer = optimizer
        self.epochs_run = 0
        self.best_val_loss = float("inf")
        self.model = model.to(self.device)
        self.processor = processor
        self.run_name = run
        self.num_epochs = config.num_epochs
        self.validate_every = config.validate_every
        self.config = config  # Store the config object
        
        # Calculate total training steps for scheduler
        num_batches_per_epoch = len(dataloader_train)
        total_steps = num_batches_per_epoch * config.num_epochs
        config.scheduler_config["num_training_steps"] = total_steps
        
        # Create checkpoint directory for this run
        self.checkpoint_dir = config.get_run_checkpoint_dir(run)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize Weights & Biases logging
        self.logger = wandb.init(
            project="caption-generator-GiT",
            name=f"experiment_{run}",
            config={
                "model_name": config.model_name,
                "seed": config.seed,
                "num_samples": config.num_samples,
                "train_split": config.train_split,
                "validate_every": config.validate_every,
                "batch_size": config.batch_size,
                "num_epochs": config.num_epochs,
                "optimizer_config": config.optimizer_config,
            }
        )

        # Find and load best checkpoint if exists
        best_checkpoint = self._find_best_checkpoint()
        if best_checkpoint:
            print(f"Found best checkpoint for run '{run}': {best_checkpoint}")
            self._load_checkpoint(best_checkpoint)
        
        # Initialize learning rate scheduler
        self.scheduler = self._create_scheduler(optimizer, config)
        
        # Initialize early stopping
        self.early_stopping_counter = 0

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
            if file.endswith(".pt"):
                checkpoint_path = os.path.join(self.checkpoint_dir, file)
                try:
                    # Load checkpoint to get validation loss
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    checkpoints.append((checkpoint_path, checkpoint["val_loss"]))
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
        
        # Load config from saved JSON file
        config_path = checkpoint.get("config_path")
        if config_path and os.path.exists(config_path):
            self.config.load_from_json(config_path)
            print(f"Loaded configuration from {config_path}")
        else:
            print("Warning: No config file found in checkpoint, using current config")
        
        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Restore training state
        self.epochs_run = checkpoint["epochs_run"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        # Log checkpoint loading information
        self.logger.log({
            "checkpoint_loaded": True,
            "checkpoint_path": checkpoint_path,
            "config_path": config_path,
            "checkpoint_epoch": checkpoint["epoch"],
            "checkpoint_val_loss": checkpoint["val_loss"],
            "checkpoint_best_val_loss": checkpoint["best_val_loss"],
            "previous_wandb_run_id": checkpoint["wandb_run_id"],
            "processor_config": checkpoint.get("processor_config", {})
        })

    def _create_scheduler(self, optimizer: torch.optim.Optimizer, config: GiTFineTuningConfig) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Creates a learning rate scheduler based on configuration.
        
        Args:
            optimizer: The optimizer to schedule
            config: Training configuration object
            
        Returns:
            Learning rate scheduler instance
        """
        scheduler_config = config.scheduler_config
        scheduler_type = scheduler_config["type"]
        
        if scheduler_type == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=scheduler_config["num_training_steps"]
            )
        elif scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config["num_training_steps"]
            )
        else:
            return torch.optim.lr_scheduler.ConstantLR(optimizer)

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
        
        # Apply mixed precision if configured
        if self.config.training_config["precision"] == "16":
            with torch.cuda.amp.autocast():
                outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                loss = outputs.loss
        else:
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=input_ids
            )
            loss = outputs.loss
            
        # Scale loss for gradient accumulation
        loss = loss / self.config.training_config["accumulate_grad_batches"]
        loss.backward()
        
        # Apply gradient clipping
        if self.config.training_config["gradient_clip_val"] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training_config["gradient_clip_val"]
            )
            
        self.optimizer.step()
        
        # Step the scheduler
        if self.scheduler is not None:
            self.scheduler.step()
            
        # Log per-batch metrics
        self.logger.log({
            "batch_loss": loss.item() * self.config.training_config["accumulate_grad_batches"],
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "batch_size": input_ids.size(0),
            "avg_attention_mask": attention_mask.float().mean().item(),
            "gradient_norm": torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                float("inf")
            ).item()
        })
        return loss.item() * self.config.training_config["accumulate_grad_batches"]

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
        
        # Save config to a separate JSON file for easier inspection
        config_path = checkpoint_path.replace(".pt", "_config.json")
        self.config.save_to_json(config_path)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "epochs_run": self.epochs_run,
            "wandb_run_id": self.logger.id,
            "run_name": self.run_name,
            "timestamp": timestamp,
            "config_path": config_path,
            "processor_config": {
                "model_name": self.config.model_name,
                "max_length": self.config.generation_config["max_length"],
                "padding": "max_length",
                "truncation": True 
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.log({
            "checkpoint_saved": True,
            "checkpoint_path": checkpoint_path,
            "config_path": config_path,
            "checkpoint_name": checkpoint_name,
            "checkpoint_epoch": epoch,
            "checkpoint_val_loss": val_loss,
            "checkpoint_timestamp": timestamp,
            "processor_config": checkpoint["processor_config"]
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
        images = []
        captions = []
        
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
                
                # Log sample predictions based on config
                if num_batches % self.config.validation_config["visualize_every_n_batches"] == 0:
                    # Generate captions for a sample
                    idx = 0
                    generated_ids = self.model.generate(
                        pixel_values=pixel_values[idx:idx+1], 
                        **self.config.generation_config
                    )
                    generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    true_caption = self.processor.batch_decode(input_ids[idx:idx+1], skip_special_tokens=True)[0]
                    
                    # Store image and captions for the grid
                    images.append(pixel_values[idx].to("cpu").permute(1,2,0).numpy())
                    captions.append(f"True: {true_caption}\nGenerated: {generated_caption}")
            
            # Create and log a single grid of images for the epoch
            if images:
                self.logger.log({
                    "validation_images": wandb.Image(
                        np.concatenate(images, axis=0),
                        caption=captions,
                    )
                })
        
        avg_val_loss = total_val_loss / num_batches
        self.logger.log({
            "validation_loss": avg_val_loss,
            "validation_batches": num_batches
        })
        return avg_val_loss
        
    def _should_stop_early(self, val_loss: float) -> bool:
        """
        Checks if training should be stopped early based on validation loss.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should be stopped, False otherwise
        """
        config = self.config.early_stopping_config
        if config["mode"] == "min":
            is_better = val_loss < (self.best_val_loss - config["min_delta"])
        else:
            is_better = val_loss > (self.best_val_loss + config["min_delta"])
            
        if is_better:
            self.best_val_loss = val_loss
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
            
        return self.early_stopping_counter >= config["patience"]
        
    def train(self) -> None:
        """
        Runs the complete training process for the configured number of epochs.
        Includes validation steps based on validate_every parameter.
        """
        try:
            for epoch in range(self.epochs_run, self.num_epochs):
                # Training step
                avg_loss = self._run_epoch(epoch)
                
                # Validation step if needed
                if (epoch + 1) % self.validate_every == 0:
                    val_loss = self._validate()
                    
                    # Update best validation loss and save checkpoint
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint(epoch, val_loss)
                        self.logger.log({
                            "best_val_loss": self.best_val_loss,
                            "epoch_of_best_val": epoch
                        })
                    
                    # Check for early stopping
                    if self._should_stop_early(val_loss):
                        print(f"Early stopping triggered after {epoch + 1} epochs")
                        self.logger.log({
                            "early_stopping_triggered": True,
                            "final_epoch": epoch + 1,
                            "best_val_loss": self.best_val_loss
                        })
                        break
                
                self.epochs_run += 1
        finally:
            self.logger.finish()

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

def train(run_name: str = 'test', config_path: Optional[str] = None) -> None:
    """
    Main training function that orchestrates the entire training process.
    
    Args:
        run_name: Name of the training run
        config_path: Optional path to configuration JSON file
    """
    # Load configuration
    config = GiTFineTuningConfig(config_path)
    
    # Load all necessary components
    model, data_train, data_val, opt, processor = load_training_artifact(config)
    
    # Create data loaders
    dataloader_train = get_dataloader(data_train, config)
    dataloader_val = get_dataloader(data_val, config)
    
    # Initialize trainer and start training
    trainer = Trainer(run_name, model, dataloader_train, dataloader_val, opt, processor, config)
    trainer.train()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train the GIT model for image captioning")
    parser.add_argument("--run_name", type=str, default="test", help="Name of the training run")
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    
    args = parser.parse_args()
    
    wandb.login()
    train(args.run_name, args.config)