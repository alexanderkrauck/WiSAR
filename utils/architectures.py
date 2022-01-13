"""
The full architectures used for predicting the final outputs.
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "04-13-2021"

from numpy.core.fromnumeric import argmax
from numpy.core.shape_base import block
from .basic_function import *
from .data import MultiViewTemporalSample, make_impossible_mask, MultiViewTemporalDataset
from .sub_architectures import ConvolutionalAutoencoderV1

from typing import List, Tuple
import torch
import numpy as np
import cv2
import json
from time import time as ti

import matplotlib.pyplot as plt
from scipy.special import expit

from abc import ABC, abstractmethod


class ScoreAnomalyDetection(ABC):
    @abstractmethod
    def score(self, sample: MultiViewTemporalSample, verbose: int) -> np.ndarray:
        """Creates a score for a single MutliViewTemporalSample
        
        Parameters
        ----------
        sample: MultiViewTemporalSample
            The sample to make a score for
        verbose: int
            The level of verboseness

        Returns
        -------
        score: np.ndarray
            The score of shape 1024x1024 of the warped image.
            The type should be np.float32 and the score for each pixel goes from 0 to 1.
        """

        raise NotImplementedError("This is abstract!")


class BasicAutoencoderAnomalyDetectionV1(ScoreAnomalyDetection):
    """A Basic approach that uses an autoencoder to detect differences. The autoencoder should take image_sizes pixel RBG images."""

    def __init__(
        self,
        pretrained_convolutional_network: ConvolutionalAutoencoderV1,
        device: str = "cpu",
        cutoff_value: float = 0.5,
        image_sizes: Tuple = (64,64),
        use_dropout: bool = False
    ):

        self.pretrained_convolutional_network = pretrained_convolutional_network.to(
            device
        )

        if use_dropout:
            self.pretrained_convolutional_network.train()
        else:
            self.pretrained_convolutional_network.eval()


        self.device = device
        self.cutoff_value = cutoff_value
        self.image_sizes = np.array(image_sizes)

        pass

    def score(self, sample: MultiViewTemporalSample, verbose: int) -> np.ndarray:

        sample_start_time = ti()
        if verbose >= 2:
            print(f"Processing sample {sample.sample_path}...")
        photos = sample.photos
        reconstructed_photos = np.empty_like(photos)

        # Step 1: Select grid from all images
        # Step 2: Put each grid-image into the autoencoder that is passed
        # Step 3: Put the reconstructed images back together (to the original time-view-structure)
        # Done here:
        if verbose >= 3:
            print(f"Using Autoencoder... ")
        autoencoder_start_time = ti()
        for timestep_idx, timestep in enumerate(photos):
            for view_idx, view in enumerate(timestep):
                grid_images = []

                grid_images = reshape_split(view, self.image_sizes)

                grid_images = (
                    np.transpose(grid_images, (0, 3, 1, 2)).astype(np.float32) / 255
                )

                torch_grid_images = torch.tensor(grid_images, device=self.device)
                reconstructed_grid_images = (
                    self.pretrained_convolutional_network(torch_grid_images)
                    .detach()
                    .cpu()
                    .numpy()
                )

                reconstructed_grid_images = np.transpose(
                    (reconstructed_grid_images * 255).astype(np.uint8), (0, 2, 3, 1)
                )

                reconstructed_view = reshape_merge(
                    reconstructed_grid_images, self.image_sizes
                )

                if verbose >= 3 and view_idx == 0 and timestep_idx == 0:
                    plt.imshow(view)
                    plt.title("The true first view")
                    plt.show()

                    plt.imshow(reconstructed_view)
                    plt.title("The reconstructed first view")
                    plt.show()
                reconstructed_photos[timestep_idx, view_idx] = reconstructed_view
        if verbose >= 3:
            print(f"Autoencoder finished. took {ti() - autoencoder_start_time:.2f}s.")

        # Step a: Take Differences between original and reconstructed
        # Step b: Differences between timesteps
        # Step c: Warp
        # Step d: Combine Timesteps
        reconstructed_photos = reconstructed_photos.astype(np.float32)/255
        photos = photos.astype(np.float32)/255
        differences = abs(photos - reconstructed_photos)
        #differences[np.amax(differences, -1) < 0.8] = 0
        if verbose >=3:
            show_photo_grid(warp_image_grid(reconstructed_photos, sample.homographies))
            plt.imshow(reconstructed_photos[3,9])
            plt.show()
            plt.imshow(photos[3,9])
            plt.show()
            plt.imshow(np.sum(differences[3,9]**2, axis=-1), cmap="gray")
            plt.show()

        if verbose >= 5:
            show_photo_grid(differences)

            show_photo_grid(warp_image_grid(differences, sample.homographies))


        if verbose >= 3:
            print(f"Warping...")
        homographies_start_time = ti()
        
        #timesteps x perspectives x height x width x color -> timesteps * perspectives x height x width x color

        flat_differences = np.reshape(differences, (-1, *differences.shape[2:]))**2
        flat_differences = (flat_differences*255).astype(np.uint8)
        flat_homographies = np.reshape(sample.homographies, (-1, *sample.homographies.shape[2:]))

        resulting_integrated = integrate_images(flat_differences, flat_homographies)




        if verbose >= 3:
            print(f"Warping finished. Took {ti() - homographies_start_time:.2f}s.")

        score_image = np.amax(
            resulting_integrated, axis=-1
        )  # max out color dimension #TODO: play around with


        return score_image.astype(np.float32)/255

    def infer(
        self,
        samples: List[MultiViewTemporalSample],
        boxing_method: str = "threshhold",
        threshhold: float = 0.5,
        verbose: int = 0,
    ):

        res_boxes = []
        start_time = ti()
        for sample in samples:
            score_image = self.score(sample, verbose)

            # Step 4: Threshhold for bounding boxes
            # Step 5: Output into the files/return values.

            score_image = (score_image * 255).astype(np.uint8)
            plt.imshow(score_image)
            plt.title("The scores of the warped image.")
            plt.show()
            score_image = cv2.GaussianBlur(score_image, (7, 7), 5)
            plt.imshow(score_image)
            plt.title("The blurred scores of the warped image.")
            plt.show()

            thresh = ((score_image >= 255 * threshhold) * 255).astype(np.uint8)
            # thresh = ((score_image >= threshhold)*255).astype(np.uint8)

            contours = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = contours[0] if len(contours) == 2 else contours[1]

            boxes = []
            for cntr in contours:
                box = np.array(cv2.boundingRect(cntr))
                boxes.append(box)
            res_boxes.append(np.array(boxes))

        return res_boxes


class BasicTimestepAnomalyDetection(ScoreAnomalyDetection):
    def __init__(self, type:str="expit"):

        self.type = type
        pass

    def score(self, sample: MultiViewTemporalSample,  verbose: int = 0) -> np.ndarray:


        view_scores = []
        for perspective in range(10):
            view_scores.append(sample.get_warped_photo(6, perspective) - sample.get_warped_photo(0, perspective))

        if self.type == "expit":
            view_scores = 1 - expit(view_scores)
            view_scores = (view_scores > 0.45).astype(np.float32)


            score = np.amax(view_scores, 0)

            #view_scores = np.array(view_scores).astype(np.float32)/255

            score = 1 - np.amax(score, -1)

        
        return score
        

    def infer(self, sample: MultiViewTemporalSample):

        photos = sample.photos

        pass

class IntegratedImagesAnomalyDetection(ScoreAnomalyDetection):
    #TODO: finish!
    def __init__(self, autoencoder):

        self.autoencoder = autoencoder
        pass

    def score(self, sample: MultiViewTemporalSample,  verbose: int = 0) -> np.ndarray:
        """
        
        Returns
        -------
        score: np.nparray
            The score is of shape 1024x1024 and contains for each pixel of the integrated image a score between 0 to 1. 1 stands for stong anomaly.
        """

        
        for timestep_idx in range(7):
            integrated_timestep_img = sample.integrate(timestep_idx)
            self.autoencoder(integrated_timestep_img)

        
        return score
        


class ScoreEnsembleAnomalyDetection:
    def __init__(
        self, score_architectures: List[ScoreAnomalyDetection], weights: List[float], min_box_size: int = 27, min_contour_area: int = 15
    ):
        """
        
        Parameters
        ----------
        score_architectures: List[ScoreAnomalyDetection]
            A list of n architectures to create scores.
        weights: List[float]
            A list of n weights.
        """

        self.score_architectures = score_architectures
        self.weights = np.array(weights).reshape(-1, 1, 1)
        self.min_box_size = min_box_size
        self.min_contour_area = min_contour_area

    def infer(self, samples: Union[List[MultiViewTemporalSample], MultiViewTemporalDataset], threshold = 0.5, verbose = 0, write_boxes_to_file: Optional[str] = None):

        res_boxes = []
        for sample in samples:
            scores = []
            for architecture in self.score_architectures:
                score = architecture.score(sample, verbose = verbose)
                if verbose >= 3:
                    plt.imshow(score, cmap='gray')
                    plt.title(f"Scores of {architecture}")
                    plt.show()
                scores.append(score)

            scores = np.array(scores)
            scores = scores * self.weights

            score = np.sum(scores, axis=0) / np.sum(self.weights) #maybe use some nonlinear thing. like maximum or softmax-ish.
            

            score = (score * 255).astype(np.uint8)


            impossible_mask = make_impossible_mask(sample)
            score[impossible_mask] = 0

            if verbose >= 3:
                plt.imshow(score, cmap='gray')
                plt.title("The combined scores")
                plt.show()
            score = cv2.GaussianBlur(score, (7, 7), 5)
            if verbose >= 3:
                plt.imshow(score, cmap='gray')
                plt.title("The combined scores blurred")
                plt.show()

            thresh = ((score >= threshold) * 255).astype(np.uint8)

            contours = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = contours[0] if len(contours) == 2 else contours[1]

            boxes = []
            for cntr in contours:
                area = cv2.contourArea(cntr)
                if area < self.min_contour_area:
                    continue

                (vertical_margin, horizonal_margin, vertical_padding, horizonal_padding) = cv2.boundingRect(cntr)

                if vertical_padding < self.min_box_size:
                    delta = self.min_box_size - vertical_padding
                    side_delta = delta // 2
                    vertical_margin = vertical_margin - side_delta
                    vertical_padding = vertical_padding + delta

                if horizonal_padding < self.min_box_size:
                    delta = self.min_box_size - horizonal_padding
                    side_delta = delta // 2
                    horizonal_margin = horizonal_margin - side_delta
                    horizonal_padding = horizonal_padding + delta

                boxes.append(np.array([vertical_margin, horizonal_margin, vertical_padding, horizonal_padding]))
            res_boxes.append(np.array(boxes))


        if write_boxes_to_file is not None:

            box_list = [el.tolist() for el in res_boxes]
            box_dict = {}
            for sample, boxes in zip(samples, box_list):
                sample_name = os.path.split(sample.sample_path)[-1]
                box_dict[sample_name] = boxes
            with open(write_boxes_to_file, "w") as fi:
                json.dump(box_dict, fi)

        return res_boxes

