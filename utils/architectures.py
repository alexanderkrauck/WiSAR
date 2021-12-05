"""
The full architectures used for predicting the final outputs.
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "04-13-2021"

from typing import List
import torch
from .data import MultiViewTemporalSample

class BasicConvolutionalAnomalyDetection:

    def __init__(self, pretrained_convolutional_network):
        
        self.pretrained_convolutional_network = pretrained_convolutional_network
        
        pass

    def infer(self, samples: List[MultiViewTemporalSample]):

        for sample in samples:
            
            photos = sample.photos

            pass #TODO: DO EVERYTHING!
    