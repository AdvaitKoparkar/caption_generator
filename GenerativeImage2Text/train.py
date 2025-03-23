"""
Main training script for fine-tuning the GIT model.
"""

import os
from GenerativeImage2Text.config import GiTFineTuningConfig
from GenerativeImage2Text.utils import load_training_artifact, get_dataloader
from GenerativeImage2Text.trainer import Trainer

def main():
    # Create a unique run name (you can modify this as needed)
    run_name = "experiment_1"
    
    # Initialize configuration
    config = GiTFineTuningConfig()
    
    # Load training artifacts (model, datasets, optimizer, processor)
    print("Loading training artifacts...")
    model, train_dataset, val_dataset, optimizer, processor = load_training_artifact(config)
    
    # Create data loaders
    print("Creating data loaders...")
    train_dataloader = get_dataloader(train_dataset, config)
    val_dataloader = get_dataloader(val_dataset, config)
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = Trainer(
        run=run_name,
        model=model,
        dataloader_train=train_dataloader,
        dataloader_val=val_dataloader,
        optimizer=optimizer,
        processor=processor,
        config=config
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    print(f"Training completed! Checkpoints saved in {config.get_run_checkpoint_dir(run_name)}")

if __name__ == "__main__":
    main() 