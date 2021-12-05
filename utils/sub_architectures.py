"""
Sub-Architectures that are used in the main architectures and only do sub-tasks.
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "04-13-2021"

import torch
from torch import nn

import os
from pathlib import Path
from datetime import datetime

# Convention Imports
from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Callable, Dict, Iterable, Optional


class AbstractTorchArchitecture(ABC):
    @abstractmethod
    def save(self, path: str, filename: str):
        """This Function should save the parameters of the model.
        
        Parameters
        ----------
        path: str
            The path where the parameter file should be stored
        filename: str
            The filename of the parameter file.
        """
        raise NotImplementedError("This is abstract!")


class ConvolutionalAutoencoderV1(nn.Module, AbstractTorchArchitecture):
    """A Convolutional Autoencoder that expects 64x64 inputs"""

    def __init__(self, p_dropout: float = 0.2):
        """
        
        Parameters
        ----------
        p_dropout : float
            The dropout probability in each of the dropouts.
        """
        super().__init__()
        self.p_dropout = p_dropout

        self.enc_conv1 = nn.Conv2d(3, 16, 3)
        self.enc_pool1 = nn.MaxPool2d((2, 2), return_indices=True)
        self.enc_conv2 = nn.Conv2d(16, 64, 5)
        self.enc_pool2 = nn.MaxPool2d((3, 3), return_indices=True)
        self.enc_conv3 = nn.Conv2d(64, 128, 4)
        self.enc_pool3 = nn.MaxPool2d((2, 2), return_indices=True)

        self.mid_conv = nn.ConvTranspose2d(128, 128, 1)

        self.dec_pool1 = nn.MaxUnpool2d((2, 2))
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, 4)
        self.dec_pool2 = nn.MaxUnpool2d((3, 3))
        self.dec_conv2 = nn.ConvTranspose2d(64, 16, 5)
        self.dec_pool3 = nn.MaxUnpool2d((2, 2))
        self.dec_conv3 = nn.ConvTranspose2d(16, 3, 3)

        self.dropout = nn.Dropout2d(p=p_dropout)

    def forward(self, x):
        x, indices1 = self.enc_pool1(self.enc_conv1(x))
        x = self.dropout(torch.relu(x))
        x, indices2 = self.enc_pool2(self.enc_conv2(x))
        x = self.dropout(torch.relu(x))
        x, indices3 = self.enc_pool3(self.enc_conv3(x))
        x = self.dropout(torch.relu(x))

        x = self.mid_conv(x)

        x = self.dec_conv1(self.dec_pool1(x, indices3))
        x = self.dropout(torch.relu(x))
        x = self.dec_conv2(self.dec_pool2(x, indices2))
        x = self.dropout(torch.relu(x))
        x = self.dec_conv3(self.dec_pool3(x, indices1))

        x = torch.sigmoid(x)
        return x

    def save(self, path: Optional[str] = None, filename: Optional[str] = None):
        """This Function saves the parameters of the model.
        
        Parameters
        ----------
        path: Optional[str]
            The path where the parameter file should be stored. If None then the path "saved_models/ConvolutionalAutoencoderV1" is used.
        filename: Optional[str]
            The filename of the parameter file. If None the timestamp is used.
        """

        if path is None:
            path = os.path.join("saved_models", "ConvolutionalAutoencoderV1")
        Path(path).mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.pt")

        save_name = os.path.join(path, filename)
        torch.save(self.state_dict(), save_name)

        print(f"Paramters saved to file '{save_name}'.")

