"""
Utility classes for loading data.
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "04-12-2021"

from .basic_function import integrate_images, draw_labels

from typing import List, Optional
import numpy as np
from PIL import Image
from torch.utils import data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import os
import json
import cv2


_photo_order = ["B05", "B04", "B03", "B02", "B01", "G01", "G02", "G03", "G04", "G05"]


class MultiViewTemporalSample:
    """The Data Structure to be used for representing single samples of the WISAR challenge.

    An instance contains a numpy array 'photos' that contains the samples,
    with the first dimension for the time (past to present) and the second one for the persepective (left to right).
    Also 'homographies' contains the loaded homography array.
    If mode is validation, also labels will be loaded into the array 'labels'.
    """

    def __init__(
        self, sample_path: str, mode: str, mask: Optional[np.ndarray] = None
    ) -> None:
        """
        
        Parameters
        ----------
        sample_path: str
            The path of the sample to be loaded into this MutliViewTemporalSample instance.
        mode: str
            Either 'train', 'validation' or 'test'. Only for 'validation' targets will be available.
        mask: Optional[np.ndarray] 
            If not None, this mask is applied.
        """

        homography_dict = json.load(
            open(os.path.join(sample_path, "homographies.json"))
        )

        self.photos = []
        self.homographies = []
        self.mask = mask
        self.mode = mode
        for timestep in range(0, 7):
            timestep_photos = []
            timestep_homographies = []
            for perspective in _photo_order:
                name = str(timestep) + "-" + perspective
                photo = np.asarray(Image.open(os.path.join(sample_path, name + ".png")))
                homography = homography_dict[name]

                if self.mask is not None:
                    photo[mask] = 0

                timestep_homographies.append(homography)
                timestep_photos.append(photo)
            timestep_photos = np.array(timestep_photos)
            timestep_homographies = np.array(timestep_homographies)
            self.photos.append(timestep_photos)
            self.homographies.append(timestep_homographies)
        self.photos = np.array(self.photos)
        self.homographies = np.array(self.homographies)

        if mode == "validation":
            self.labels = np.array(
                json.load(open(os.path.join(sample_path, "labels.json")))
            )

    def show_photo_grid(self):
        """Show a photo grid of all photos in the sample"""

        fig, ax = plt.subplots(7, 10, figsize=(15, 10))
        for row, timeframe in enumerate(self.photos):
            for col, perspective in enumerate(timeframe):
                ax[row, col].imshow(perspective)
                ax[row, col].axis("off")
                ax[row, col].set_xticklabels([])
                ax[row, col].set_yticklabels([])

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()

    def integrate(self, timestep: int = 0) -> np.ndarray:
        """Integrate the Images of this sample for a given timestep
        
        Parameters
        ----------
        timestep: int
            The timestep (starting at 0) where the pictures should be integrated over.
        """

        integrated_image = integrate_images(
            images=self.photos[timestep],
            homographies=self.homographies[timestep],
            mask=self.mask,
        )

        return integrated_image

    def draw_labels(
        self, labels: Optional[np.ndarray] = None, on_integrated: bool = False
    ):
        """Draws images with bounding boxes for predictions
        
        Parameters
        ----------
        labels: Optional[np.ndarray]
            If not None, then those are the bounding boxes in an numpy array.
            The first dimension corresponds to the number of labels and in the second dimension should, as provided, be
            margin x, margin y, size x, size y.
            If None, then the bounding boxes of the validation data are used.
        on_integrated: bool
            If True, then the bounding boxes are drawn on the integrated images of timestep 3 (the center timestep).
            If False, then the bounding boxes are drawn on the center camera and center timestep (3_B01).
        """

        if on_integrated:
            image = self.integrate(timestep=3)
        else:
            image = self.photos[3, 4].copy()  # the center image = 3_B01

        if labels is None:
            assert self.mode == "validation"
            labels = self.labels

        draw_labels(image, labels)

    def get_warped_photo(self, timestep: int, perspective: int) -> np.ndarray:
        """Get a specific warped photo from this sample.

        Parameters
        ----------
        timestep: int
            The timestep of the wanted image (from 0 to 6)
        perspective: int
            The perspective of the wanted image (from 0 to 9)

        Returns
        -------
        warped_photo: np.ndarray
            The wanted photo warped using the homographies of this sample.
        """

        photo = self.photos[timestep, perspective]
        homography = self.homographies[timestep, perspective]

        warped_photo = cv2.warpPerspective(photo, homography, photo.shape[:2])

        return warped_photo


class MultiViewTemporalDataset(Dataset):
    """Dataset class that should be used for loading the provided data.
    
    The ImageDataset loads all samples into the memory and stores each sample in a MutliViewTemporalSample instance.
    """

    def __init__(
        self, data_path: str = "data", mode: str = "train", apply_mask: bool = True
    ):
        """

        Parameters
        ----------
        data_path: str
            The path where all data is included. This means the folder should contain train, test and validation folders and the mask.
        mode: str
            Either 'train', 'validation' or 'test'. Only for 'validation' targets will be available.
        apply_mask: bool
            If True, then the supplied mask will be applied on all pictures.
        #TODO: Lazy loading might be necessary as the data is rather large!
        """

        mode = mode.lower()
        assert mode in ["train", "test", "validation"]

        self.path = os.path.join(data_path, mode)

        if apply_mask:
            mask = ~np.asarray(Image.open(os.path.join("data", "mask.png")), dtype=bool)
        else:
            mask = None

        self.samples = [
            MultiViewTemporalSample(os.path.join(self.path, s), mode, mask=mask)
            for s in os.listdir(self.path)
            if os.path.isdir(os.path.join(self.path, s))
        ]
        self.samples = np.array(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


class GridCutoutDataset(MultiViewTemporalDataset):
    """Dataset for training autoencoders on single subimages of custom size
    
        
    """

    def __init__(
        self,
        cutout_shape: int = 64,
        randomize: bool = False,
        data_path: str = "data",
        mode: str = "train",
        apply_mask: bool = True,
    ):
        # TODO think about removing primarely black images!
        """

        Parameters
        ----------
        cutout_shape: int
            Each image returned will have a shape of [cutout_shape, cutout_shape, 3]
        randomize: bool TODO
            If there is freedom w.r.t. to the cutout_shape (i.e. if cutout_shape is not a divisor of 1024)
            then the remaining pixels are randomly chosen such that everything is used.
        data_path: str
            The path where all data is included. This means the folder should contain train, test and validation folders and the mask.
        mode: str
            Either 'train', 'validation' or 'test'. Only for 'validation' targets will be available.
        apply_mask: bool
            If True, then the supplied mask will be applied on all pictures.
        """

        super().__init__(data_path=data_path, mode=mode, apply_mask=apply_mask)

        samples = []
        for sample in self.samples:
            sample = sample.photos
            sample = sample.reshape(-1, *sample.shape[-3:])
            samples.append(sample)

        self.samples = np.concatenate(samples)
        self.randomize = randomize

        self.cutout_shape = cutout_shape
        self.n_row_subindices = 1024 // cutout_shape
        self.n_total_subindices = self.n_row_subindices ** 2

    def __len__(self):
        return len(self.samples) * self.n_total_subindices

    def __getitem__(self, index: int):
        subindex = index % self.n_total_subindices
        hard_index = index // self.n_total_subindices

        sample = self.samples[hard_index]
        col = subindex % self.n_row_subindices
        row = subindex // self.n_row_subindices

        # TODO: add optional randomization somehow
        cut_sample = sample[
            row * self.cutout_shape : (row + 1) * self.cutout_shape,
            col * self.cutout_shape : (col + 1) * self.cutout_shape,
        ]

        cut_sample = np.transpose(cut_sample, (2, 0, 1)).astype(np.float32) / 255

        return cut_sample

