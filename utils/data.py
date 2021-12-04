"""
Utility classes for loading data.
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "04-12-2021"

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import os
import json
import cv2


class MultiViewTemporalSample:
    """The Data Structure to be used for representing single samples of the WISAR challenge.

    An instance contains a numpy array 'photos' that contains the samples,
    with the first dimension for the time (past to present) and the second one for the persepective (left to right).
    Also 'homographies' contains the loaded homography array.
    If mode is validation, also labels will be loaded into the array 'labels'.
    """

    def __init__(self, sample_path: str, mode:str, apply_mask: bool = False) -> None:
        """
        
        Parameters
        ----------
        sample_path: str
            The path of the sample to be loaded into this MutliViewTemporalSample instance.
        mode: str
            Either 'train', 'validation' or 'test'. Only for 'validation' targets will be available.
        apply_mask: bool TODO: implement this!
            Whether the mask should be applied to all images of the sample.
        """

        self.photos = []
        for timestep in range(0, 7):
            timestep_photos = []
            for prespective in [
                "B05",
                "B04",
                "B03",
                "B02",
                "B01",
                "G01",
                "G02",
                "G03",
                "G04",
                "G05",
            ]:
                name = str(timestep) + "-" + prespective + ".png"
                im = Image.open(os.path.join(sample_path, name))
                # TODO: apply first preprocesseing steps here (e.g. using the mask)
                timestep_photos.append(im)
            timestep_photos = np.array(timestep_photos)
            self.photos.append(timestep_photos)
        self.photos = np.array(self.photos)
        self.homography = json.load(
            open(os.path.join(sample_path, "homographies.json"))
        )

        if mode == "validation":
            self.labels = json.load(
                open(os.path.join(sample_path, "labels.json"))
            )

    def show_photo_grid(self):
        """Show a photo grid of all photos in the sample"""

        fig, ax = plt.subplots(7, 10, figsize=(15, 10))
        for row, timeframe in enumerate(self.photos):
            for col, perspective in enumerate(timeframe):
                ax[row, col].imshow(np.array(perspective))
                ax[row, col].axis("off")
                ax[row, col].set_xticklabels([])
                ax[row, col].set_yticklabels([])

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()

    def warp(self, image_id="0-B01"):
        #TODO!!! code structure is prelininary and can be removed!!
        homo = np.array(self.homography[image_id])

        warped = cv2.warpPerspective(
            self.photos[0][5], homo, self.photos[0][5].shape[:2]
        )
        return warped


class ImageDataset(Dataset):
    """Dataset class that should be used for loading the provided data.
    
    The ImageDataset loads all samples into the memory and stores each sample in a MutliViewTemporalSample instance.
    """

    def __init__(self, data_path: str = "data", mode: str = "train"):
        """

        Parameters
        ----------
        data_path: str
            The path where all data is included. This means the folder should contain train, test and validation folders and the mask.
        mode: str
            Either 'train', 'validation' or 'test'. Only for 'validation' targets will be available.
        #TODO: Lazy loading might be necessary as the data is rather large!
        """

        mode = mode.lower()
        assert mode in ["train", "test", "validation"]

        self.path = os.path.join(data_path, mode)

        self.samples = [
            MultiViewTemporalSample(os.path.join(self.path, s))
            for s in os.listdir(self.path)
            if os.path.isdir(os.path.join(self.path, s))
        ]
        self.samples = np.array(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index:int):
        return self.samples[index]


#class Pytorch_Dataloader(DataLoader):
    #TODO: Code was useless because the DataLoader should get a Dataset instance and only load the samples there, i.e. minibatch them etc...
    #We might need a custom 'collate_fn' depending on the architecture we choose.

