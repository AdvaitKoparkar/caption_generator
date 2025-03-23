"""
Trainer module for the Generative Image to Text (GIT) model training.
"""

import os
import torch
import wandb
from tqdm import tqdm
from typing import Dict, Any, Optional
import datetime

from transformers import AutoProcessor
from .config import GiTFineTuningConfig

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

        # Initialize learning rate scheduler
        self.scheduler = self._create_scheduler(optimizer, config)
        
        # Initialize early stopping counter
        self.early_stopping_counter = 0

        # Find and load best checkpoint if exists
        best_checkpoint = self._find_best_checkpoint()
        if best_checkpoint:
            print(f"Found best checkpoint for run '{run}': {best_checkpoint}")
            self._load_checkpoint(best_checkpoint)

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
            "config_path": config_path,  # Store path to config file
            "processor_config": {
                "model_name": self.config.model_name,
                "max_length": self.config.generation_config["max_length"],
                "padding": "max_length",  # This is fixed in our dataset
                "truncation": True  # This is fixed in our dataset
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
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.dataloader_train, desc=f"Training Epoch {epoch}")
        for batch in progress_bar:
            loss = self._step(batch)
            total_loss += loss
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss:.4f}",
                "avg_loss": f"{total_loss/num_batches:.4f}"
            })
            
        avg_loss = total_loss / num_batches
        self.logger.log({
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_batches": num_batches
        })
        return avg_loss

    def _validate(self) -> float:
        """
        Runs validation on the validation dataset.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_val_loss = 0.0
        num_batches = 0
        images = []  # List to store images for the grid
        captions = []  # List to store captions for the grid
        
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
                    idx = torch.randint(0, (pixel_values.size(0),))
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
                        images,
                        caption=captions,
                        key=f"validation_grid_epoch{self.epochs_run}"
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