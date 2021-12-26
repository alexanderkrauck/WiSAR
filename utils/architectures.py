"""
The full architectures used for predicting the final outputs.
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "04-13-2021"

from numpy.core.fromnumeric import argmax
from numpy.core.shape_base import block
from .basic_function import integrate_images, reshape_split, reshape_merge
from .data import MultiViewTemporalSample
from .sub_architectures import ConvolutionalAutoencoderV1

from typing import List
import torch
import numpy as np
import cv2
from time import time as ti

import matplotlib.pyplot as plt

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

    def infer(self, samples: List[MultiViewTemporalSample], boxing_method: str = "n_top", n_top: int = 10, threshhold:float = 0.5, verbose:int = 0):

        res_boxes = []
        start_time = ti()
        for sample in samples:
            sample_start_time = ti()
            if verbose >= 2: print(f"Processing sample {sample.sample_path}...")
            photos = sample.photos
            reconstructed_photos = np.empty_like(photos)

            # Step 1: Select grid from all images
            # Step 2: Put each grid-image into the autoencoder that is passed
            # Step 3: Put the reconstructed images back together (to the original time-view-structure)
            # Done here:
            if verbose >= 3: print(f"Using Autoencoder... ")
            autoencoder_start_time = ti()
            for timestep_idx, timestep in enumerate(photos):
                for view_idx, view in enumerate(timestep):
                    grid_images = []

                    grid_images = reshape_split(view, np.array([64,64]))


                    grid_images = (
                        np.transpose(grid_images, (0, 3, 1, 2)).astype(np.float32) / 255
                    )

                    torch_grid_images = torch.tensor(grid_images, device=self.device)
                    reconstructed_grid_images = self.pretrained_convolutional_network(torch_grid_images).detach().cpu().numpy()
                    
                    reconstructed_grid_images = np.transpose(
                        (reconstructed_grid_images * 255).astype(np.int16), (0, 2, 3, 1)
                    )

                    reconstructed_view = reshape_merge(reconstructed_grid_images, np.array([64,64]))


                    if verbose >= 5 and view_idx == 0 and timestep_idx == 0:

                        plt.imshow(view)
                        plt.show()
                        plt.imshow(reconstructed_view)
                        plt.show()
                    reconstructed_photos[timestep_idx, view_idx] = reconstructed_view
            if verbose >= 3: print(f"Autoencoder finished. took {ti() - autoencoder_start_time:.2f}s.")


            # Step a: Take Differences between original and reconstructed
            # Step b: Differences between timesteps
            # Step c: Warp
            # Step d: Combine Timesteps
            differences = abs(photos - reconstructed_photos)

            integrated = []

            if verbose >= 3: print(f"Warping...")
            homographies_start_time = ti()
            for timestep_differences, homographie in zip(
                differences, sample.homographies
            ):
                integrated.append(
                    integrate_images(timestep_differences, homographie, sample.mask)
                )

            integrated = np.array(integrated).astype(np.float32)/255

            squared_integrated = np.square(integrated)
            resulting_integrated = np.mean(squared_integrated, axis=0)
            if verbose >= 3: print(f"Warping finished. Took {ti() - homographies_start_time:.2f}s.")


            score_image = np.mean(resulting_integrated, axis=-1)#average out color dimension



            # Step 4: Threshhold for bounding boxes
            # Step 5: Output into the files/return values.
            if boxing_method == "threshhold":

                score_image = (score_image * 255).astype(np.uint8)
                plt.imshow(score_image)
                plt.show()
                score_image = cv2.GaussianBlur(score_image, (3,3), 2)
                plt.imshow(score_image)
                plt.show()

                thresh = ((score_image >= 255 * threshhold) * 255).astype(np.uint8)
                #thresh = ((score_image >= threshhold)*255).astype(np.uint8)
                
                contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[0] if len(contours) == 2 else contours[1]

                boxes = []
                for cntr in contours:
                    box = np.array(cv2.boundingRect(cntr))
                    boxes.append(box)
                res_boxes.append(np.array(boxes))
            
            if boxing_method == "n_top":

                values = []
                boxes = []
                for i in range(250):
                    for e in range(250):
                        value = np.mean(resulting_integrated[i * 4: (i+6)*4, e * 4: (e+6)*4])
                        values.append(value)
                        box = np.array([e*4, i*4, 6*4, 6*4])#switch axes here
                        boxes.append(box)

                values = np.array(values)
                boxes = np.array(boxes)

            
                top_indices = np.argsort(values)[-n_top:]
                top_boxes = boxes[top_indices]


                res_boxes.append(top_boxes)

        return res_boxes






class BasicTimestepAnomalyDetection:
    def __init__(self):

        pass

    def infer(self, sample: MultiViewTemporalSample):

        photos = sample.photos

        pass

