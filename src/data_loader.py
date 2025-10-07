"""
Data loader for sensor datasets
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset


class SensorDataLoader:
    """
    Data loader for industrial sensor datasets

    This class handles loading, preprocessing, and splitting of sensor data
    for training digital twin models.
    """

    def __init__(self, data_path=None, df=None):
        """
        Initialize the data loader

        Args:
            data_path (str, optional): Path to CSV file containing sensor data
            df (pd.DataFrame, optional): Pre-loaded DataFrame
        """
        if df is not None:
            self.df = df.copy()
        elif data_path is not None:
            self.df = pd.read_csv(data_path)
        else:
            self.df = None

        self.scaler_X = None
        self.scaler_y = None

    def load_data(self, data_path):
        """
        Load data from CSV file

        Args:
            data_path (str): Path to CSV file

        Returns:
            pd.DataFrame: Loaded data
        """
        self.df = pd.read_csv(data_path)
        return self.df

    def get_available_signals(self):
        """
        Get list of available sensor signals

        Returns:
            list: List of signal names
        """
        if self.df is None:
            return []

        cols = self.df.columns.tolist()

        # Remove timestamp columns
        if cols and (cols[0].startswith('2025') or
                     'timestamp' in cols[0].lower() or
                     'time' in cols[0].lower()):
            cols = cols[1:]

        return cols

    def prepare_data(self, boundary_signals, target_signals,
                     test_size=0.2, val_size=0.2, random_state=42):
        """
        Prepare and split data for training

        Args:
            boundary_signals (list): List of boundary condition signal names
            target_signals (list): List of target signal names
            test_size (float): Proportion of data for testing. Default: 0.2
            val_size (float): Proportion of training data for validation. Default: 0.2
            random_state (int): Random seed for reproducibility. Default: 42

        Returns:
            dict: Dictionary containing train/val/test splits and scalers
        """
        if self.df is None:
            raise ValueError("No data loaded. Please load data first.")

        # Extract features and targets
        X = self.df[boundary_signals].values
        y = self.df[target_signals].values

        # Fit scalers
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=test_size, random_state=random_state
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=random_state
        )

        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y
        }

    def create_dataloaders(self, X_train, y_train, X_val, y_val,
                          batch_size=64, shuffle=True):
        """
        Create PyTorch DataLoaders

        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation targets
            batch_size (int): Batch size. Default: 64
            shuffle (bool): Whether to shuffle training data. Default: True

        Returns:
            tuple: (train_loader, val_loader)
        """
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def inverse_transform_predictions(self, y_pred_scaled):
        """
        Inverse transform scaled predictions back to original scale

        Args:
            y_pred_scaled (np.ndarray): Scaled predictions

        Returns:
            np.ndarray: Predictions in original scale
        """
        if self.scaler_y is None:
            raise ValueError("Scaler not fitted. Please prepare data first.")

        return self.scaler_y.inverse_transform(y_pred_scaled)

    def get_data_info(self):
        """
        Get information about loaded data

        Returns:
            dict: Data information including shape and column names
        """
        if self.df is None:
            return {"status": "No data loaded"}

        return {
            "shape": self.df.shape,
            "columns": self.df.columns.tolist(),
            "num_samples": len(self.df),
            "num_features": len(self.df.columns)
        }
