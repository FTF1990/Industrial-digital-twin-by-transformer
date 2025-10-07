"""
Trainer module for V1 and V4 models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm


class ModelTrainer:
    """
    Unified trainer for V1 and V4 Transformer models

    This class handles the training loop, validation, early stopping,
    and learning rate scheduling for digital twin models.
    """

    def __init__(self, model, device='cuda', config=None):
        """
        Initialize the trainer

        Args:
            model (nn.Module): The model to train (V1 or V4)
            device (str): Device to use for training. Default: 'cuda'
            config (dict, optional): Training configuration
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Default configuration
        self.config = {
            'lr': 0.001,
            'weight_decay': 1e-5,
            'epochs': 100,
            'batch_size': 64,
            'grad_clip': 1.0,
            'early_stop_patience': 25,
            'scheduler_patience': 10,
            'scheduler_factor': 0.5
        }

        if config:
            self.config.update(config)

        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None

    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=self.config['scheduler_patience'],
            factor=self.config['scheduler_factor']
        )

    def train_epoch(self, train_loader):
        """
        Train for one epoch

        Args:
            train_loader (DataLoader): Training data loader

        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(batch_X)
            loss = self.criterion(predictions, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                          max_norm=self.config['grad_clip'])
            self.optimizer.step()

            train_loss += loss.item()

        return train_loss / len(train_loader)

    def validate(self, val_loader):
        """
        Validate the model

        Args:
            val_loader (DataLoader): Validation data loader

        Returns:
            float: Average validation loss
        """
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                predictions = self.model(batch_X)
                val_loss += self.criterion(predictions, batch_y).item()

        return val_loss / len(val_loader)

    def train(self, train_loader, val_loader, verbose=True):
        """
        Complete training loop

        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            verbose (bool): Whether to print training progress. Default: True

        Returns:
            dict: Training history and best model state
        """
        if self.optimizer is None:
            self.setup_optimizer()

        patience_counter = 0

        if verbose:
            print(f"Starting training on {self.device}")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print("=" * 80)

        for epoch in range(self.config['epochs']):
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validation
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)

            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                status_marker = "⭐"
            else:
                patience_counter += 1
                status_marker = "  "

            # Print progress
            if verbose:
                print(f"{status_marker} Epoch [{epoch+1:3d}/{self.config['epochs']:3d}] | "
                      f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                      f"Best: {self.best_val_loss:.6f} | LR: {current_lr:.2e} | "
                      f"Patience: {patience_counter}/{self.config['early_stop_patience']}")

            # Early stopping
            if patience_counter >= self.config['early_stop_patience']:
                if verbose:
                    print(f"\n🛑 Early stopping at epoch {epoch+1}")
                break

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        if verbose:
            print("=" * 80)
            print(f"✅ Training completed! Best validation loss: {self.best_val_loss:.6f}")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'best_model_state': self.best_model_state
        }

    def save_model(self, path):
        """
        Save model checkpoint

        Args:
            path (str): Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }, path)

    def load_model(self, path):
        """
        Load model checkpoint

        Args:
            path (str): Path to the saved model
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.best_val_loss = checkpoint['best_val_loss']
