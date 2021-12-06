"""
The full architectures used for predicting the final outputs.
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "04-13-2021"

from numpy.core.fromnumeric import argmax
from .basic_function import integrate_images
from .data import MultiViewTemporalSample
from .sub_architectures import ConvolutionalAutoencoderV1

from typing import List
import torch
import numpy as np


class BasicAutoencoderAnomalyDetectionV1:
    """A Basic approach that uses an autoencoder to detect differences. The autoencoder should take 64x64 pixel RBG images."""

    def __init__(
        self,
        pretrained_convolutional_network: ConvolutionalAutoencoderV1,
        device: str = "cpu",
        cutoff_value: float = 0.5
    ):

        self.pretrained_convolutional_network = pretrained_convolutional_network.to(
            device
        )
        self.device = device
        self.cutoff_value = cutoff_value

        pass

    def infer(self, samples: List[MultiViewTemporalSample], n_top: int = 10):

        res_boxes = []
        for sample in samples:

            photos = sample.photos
            reconstructed_photos = np.empty_like(photos)

            # Step 1: Select grid from all images
            # Step 2: Put each grid-image into the autoencoder that is passed
            # Step 3: Put the reconstructed images back together (to the original time-view-structure)
            # Done here:
            for timestep_idx, timestep in enumerate(photos):
                for view_idx, view in enumerate(timestep):
                    grid_images = []
                    reconstructed_view = np.empty_like(view)
                    for row in range(16):
                        for col in range(16):
                            grid_images.append(
                                view[
                                    row * 64 : (row + 1) * 64, col * 64 : (col + 1) * 64
                                ]
                            )

                    grid_images = np.array(grid_images)
                    grid_images = (
                        np.transpose(grid_images, (0, 3, 1, 2)).astype(np.float32) / 255
                    )

                    grid_images = torch.tensor(grid_images, device="cpu")
                    reconstructed_grid_images = (
                        self.pretrained_convolutional_network(grid_images)
                        .detach().cpu()
                        .numpy()
                    )
                    reconstructed_grid_images = np.transpose(
                        (reconstructed_grid_images * 255).astype(np.int16), (0, 2, 3, 1)
                    )

                    i = 0
                    for row in range(16):
                        for col in range(16):
                            reconstructed_view[
                                row * 64 : (row + 1) * 64, col * 64 : (col + 1) * 64
                            ] = reconstructed_grid_images[i]
                            i += 1
                    reconstructed_photos[timestep_idx, view_idx] = reconstructed_view


            # Step a: Take Differences between original and reconstructed
            # Step b: Differences between timesteps
            # Step c: Warp
            # Step d: Combine Timesteps
            differences = abs(photos - reconstructed_photos)

            integrated = []
            for timestep_differences, homographie in zip(
                differences, sample.homographies
            ):
                integrated.append(
                    integrate_images(timestep_differences, homographie, sample.mask)
                )

            integrated = np.array(integrated).astype(np.float32)/255

            squared_integrated = np.square(integrated)
            resulting_integrated = np.mean(squared_integrated, axis=0)


            # Step 4: Threshhold for bounding boxes
            # Step 5: Output into the files/return values.
            values = []
            boxes = []
            for i in range(250):
                for e in range(250):
                    value = np.mean(resulting_integrated[i * 4: (i+6)*4, e * 4: (e+6)*4])
                    values.append(value)
                    box = np.array([e*4, i*4, 6*4, 6*4])#switch axes here
                    boxes.append(box)

            top_indices = np.argsort(np.array(values))[-n_top:]
            top_boxes = np.array(boxes)[top_indices]
            res_boxes.append(top_boxes)

        return res_boxes
        # TODO:





class BasicTimestepAnomalyDetection:
    def __init__(self):

        pass

    def infer(self, sample: MultiViewTemporalSample):

        photos = sample.photos

        pass

