"""
The full architectures used for predicting the final outputs.
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "04-13-2021"

from typing import List
from numpy import dtype
import torch
from .data import MultiViewTemporalSample


class BasicConvolutionalAnomalyDetection:
    def __init__(self, pretrained_convolutional_network):

        self.pretrained_convolutional_network = pretrained_convolutional_network

        pass

    def infer(self, samples: List[MultiViewTemporalSample]):

        for sample in samples:

            photos = sample.photos
            
            #TODO:
            #Step 1: Select grid from all images
            #Step 2: Put each grid-image into the autoencoder that is passed
            #Step 3: Put the reconstructed images back together (to the original time-view-structure)

            #Step a: Take Differences between original and reconstructed
            #Step b: Differences between timesteps
            #Step c: Warp
            #Step d: Combine Timesteps

            #Step 4: Threshhold for bounding boxes
            #Step 5: Output into the files/return values.


            pass

class BasicTimestepAnomalyDetection:
    def __init__(self):

        pass

    def infer(self, sample: MultiViewTemporalSample):

        photos = sample.photos

        pass

