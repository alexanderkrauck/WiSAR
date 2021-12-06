from typing import List, Optional, Union
import numpy as np
import cv2
import matplotlib.pyplot as plt


def integrate_images(
    images: np.ndarray, homographies: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Integrate time given images using the given homographies and the mask.
    
    Parameters
    ----------
    images: np.ndarray
        The images to integrate. The array should have shape [m,n,n,3], where n is the size of the image and m is the number of images.
    homographies: np.ndarray
        The homographies for the images. The array should be of shape [m, 3, 3].
    mask: np.ndarray
        The mask for the images with shape [n,n], where 1s are for the spots that should be removed.

    Returns
    -------
    integrated_image: np.ndarray
        The integrated image of shape [n,n,3]
    """

    ov_mask = ~mask

    integrated_image = np.zeros((1024, 1024, 3))

    for photo, homography in zip(images, homographies):

        warped_image = cv2.warpPerspective(photo, homography, photo.shape[:2])

        ov_mask = np.where(np.sum(warped_image, axis=-1) > 0, ov_mask, False)
        integrated_image += warped_image

    integrated_image[~ov_mask] = 0
    integrated_image /= 10

    return np.uint8(integrated_image)

def draw_labels(image: np.ndarray, labels: Union[np.ndarray, List[np.ndarray]], plot_result: bool = True):
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
        color = [0,0,0]
        if idx<3:
            color[idx] = 255
        color = tuple(color)
        for label in labels_type:
            image = cv2.rectangle(
                image,
                (label[0], label[1]),
                (label[0] + label[2], label[1] + label[3]),
                color,
                5
            )

    if(plot_result):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.show()

    return image