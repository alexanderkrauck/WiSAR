import numpy as np
import cv2


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
