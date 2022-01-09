from typing import List, Optional, Union
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os


mask__ = ~np.asarray(Image.open(os.path.join("data", "mask.png")), dtype=bool)


def integrate_images(
    images: np.ndarray, homographies: np.ndarray
) -> np.ndarray:
    """Integrate time given images using the given homographies and the mask.
    
    Parameters
    ----------
    images: np.ndarray
        The images to integrate. The array should have shape [m,n,n,3], where n is the size of the image and m is the number of images.
    homographies: np.ndarray
        The homographies for the images. The array should be of shape [m, 3, 3].

    Returns
    -------
    integrated_image: np.ndarray
        The integrated image of shape [n,n,3]
    """

    ov_mask = ~mask__
    ov_mask = ov_mask.astype(np.uint8)


    integrated_image = np.zeros(images.shape[1:], dtype=np.int16)
    integrated_masks = np.zeros_like(ov_mask) #TODO if images are not 1024x1024 then this is a problem here

    for photo, homography in zip(images, homographies):

        warped_image = cv2.warpPerspective(photo, homography, photo.shape[:2])
        warped_mask = cv2.warpPerspective(ov_mask, homography, ov_mask.shape[:2])

        integrated_image += warped_image
        integrated_masks += warped_mask > 0


    integrated_masks = np.where(integrated_masks == 0, 1, integrated_masks)
    integrated_image = integrated_image / np.expand_dims(integrated_masks, -1)

    return np.uint8(integrated_image)


def draw_labels(
    image: np.ndarray,
    labels: Union[np.ndarray, List[np.ndarray]],
    plot_result: bool = True,
):
    """Given an image, draw the labels on top of the image and alternatively show the result.
    
    Parameters
    ----------
    image: np.ndarray
        The image where the labels should be drawn on.
    labels: Union[np.ndarray, List[np.ndarray]]
        The labels that should be drawn. If a numpy array, the first dimension corresponds to the number of labels
        and in the second dimension should, as provided, be margin x, margin y, size x, size y.
        If a list of np.ndarrays, then each of the np.ndarray will have the labels in a different color.
    plot_result: bool
        If true then the results are ploted.

    Returns
    -------
    drawn_image: np.ndarray
        The image with the bounding boxes drawn in the same shape of image.
    """

    if isinstance(labels, np.ndarray):
        labels = [labels]

    for idx, labels_type in enumerate(labels):
        color = [0, 0, 0]
        if idx < 3:
            color[idx] = 255
        color = tuple(color)
        for label in labels_type:
            image = cv2.rectangle(
                image,
                (label[0], label[1]),
                (label[0] + label[2], label[1] + label[3]),
                color,
                5,
            )

    if plot_result:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.show()

    return image


def reshape_split(array: np.ndarray, box_size: np.ndarray):
    """Returns a 'boxed' version of the input.
    
    Usually an image as input is expected (a numpy array of shape heigth x width x channels).

    Parameters
    ----------
    array: np.ndarray
        The input to be 'boxed'.
    box_size: np.ndarray
        The size of the boxes to be made with (heigth, width).

    Returns
    -------
    A flattened (along the grid dimensions) array of shape (n, box_size[0], box_size[1], channels).
    n is the number of extracted boxes.
    """

    assert array.shape[0] % box_size[0] == 0
    assert array.shape[1] % box_size[1] == 0

    arr = array.reshape(
        array.shape[0] // box_size[0],
        box_size[0],
        array.shape[1] // box_size[1],
        box_size[1],
        array.shape[-1],
    )
    arr = arr.swapaxes(1, 2)

    arr = arr.reshape(-1, *box_size, array.shape[-1])

    return arr


def reshape_merge(
    array: np.ndarray, box_size, original_shape=np.array([1024, 1024, 3])
):
    """The inverse function of reshape_split
    
    Takes a (flat) grid of images and puts it back to the original image

    Parameters
    ----------
    array: np.ndarray
        The boxes to be merged of shape (n, box_size[0], box_size[1], channels).
    box_size: np.ndarray
        The size of the boxes to be assumed (heigth, width).
    original_shape: np.ndarray
        The original shape of the image that was inputted in reshape_split.

    Returns
    -------
    The merged image.
    """

    arr = array.reshape(
        original_shape[0] // box_size[0],
        original_shape[1] // box_size[1],
        *array.shape[1:]
    )
    arr = arr.swapaxes(1, 2)
    arr = arr.reshape(original_shape)

    return arr


def preprocess_image(
    image: Union[np.ndarray, str],
    use_mask: bool = False,
    equalize_hist: bool = False,
    crop_black: bool = False,
) -> np.ndarray:
    """This function can preprocess images appropriatly
    
    Parameters
    ----------
    image:Union[np.ndarray, str]
        If np.ndarray, then this is the image to be preprocessed. 
        If str, then it is assumed that this is the image path and it will be loaded and thereafter be preprocess.
    use_mask: bool
        If true, the provided mask will be applied on the image
    equalize_hist: bool
        If true, then the color distribution of each image will be corrected
    crop_blac: bool
        If true, then the black bars will be removed from the image, then the image will have the shape 592x1024, otherwise 1024x1024.
    """

    if isinstance(image, str):
        image = np.array(Image.open(image))

    if use_mask:
        image[mask__] = 0

    if crop_black:
        image = image[216:-216]

    if equalize_hist:
        for channel in range(3):
            image[..., channel] = cv2.equalizeHist(image[..., channel])

    return image

def show_photo_grid(photo_grid: np.ndarray):
    """Show a photo grid of time and perspective dimensions"""

    n_timesteps = photo_grid.shape[0]
    n_perspectives = photo_grid.shape[1]

    _, ax = plt.subplots(n_timesteps, n_perspectives, figsize=(15, 10))
    for row, timeframe in enumerate(photo_grid):
        for col, perspective in enumerate(timeframe):
            ax[row, col].imshow(perspective)
            ax[row, col].axis("off")
            ax[row, col].set_xticklabels([])
            ax[row, col].set_yticklabels([])

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

def warp_image(image: np.ndarray, homography: np.ndarray):

    warped_photo = cv2.warpPerspective(image, homography, image.shape[:2])

    return warped_photo

def warp_image_grid(image_grid: np.ndarray, homographies: np.ndarray):
    
    warped_image_grid = np.empty_like(image_grid) 

    for timestep in range(image_grid.shape[0]):
        for perspective in range(image_grid.shape[1]):
            warped_image_grid[timestep, perspective] = warp_image(image_grid[timestep, perspective], homographies[timestep, perspective])

    return warped_image_grid