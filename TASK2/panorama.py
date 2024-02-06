#####TUWIEN - WS2023 CV: Task2 - Image Stitching
#####*********+++++++++*******++++INSERT GROUP NO. 4
from typing import List
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import cv2


def get_simple(images: List[np.ndarray], width: int, height: int, H: List[np.ndarray], T: np.ndarray) -> np.ndarray:
    """
    Stitch the final panorama with the calculated panorama extents
    by transforming every image to the same coordinate system as the center image. Use the dot product
    of the translation matrix 'T' and the homography per image 'H' as transformation matrix.
    HINT: cv2.warpPerspective(..), cv2.addWeighted(..)
    
    Parameters
    ----------
    images : List[np.ndarray]
        list of images
    width : int
        width of panorama (in pixel)
    height : int
        height of panorama (in pixel)
    H : List[np.ndarray]
        list of homographies to center image ([number_of_images x 3 x 3])
    T : np.ndarray
        translation matrix for panorama ([3 x 3])

    Returns
    ---------
    np.ndarray
        (result) panorama image ([height x width x 3])
    """
    
    # student_code start

    trans_images = []
    for i in range(len(images)):
        trans_img = cv2.warpPerspective(images[i], T.dot(H[i]), (width, height))
        trans_images.append(trans_img)

    result = np.zeros(trans_images[0].shape)
    for i in range(len(images)):
        result = cv2.addWeighted(result, 1, trans_images[i], 1, 0, dtype=cv2.CV_64F)
    result = result/255

    # student_code end
        
    return result


def get_blended(images: List[np.ndarray], width: int, height: int, H: List[np.ndarray], T: np.ndarray) -> np.ndarray:
    """
    Use the equation from the assignment description to overlay transformed
    images by blending the overlapping colors with the respective alpha values
    HINT: ndimage.distance_transform_edt(..)
    
    Parameters
    ----------
    images : List[np.ndarray]
        list of images
    width : int
        width of panorama (in pixel)
    height : int
        height of panorama (in pixel)
    H : List[np.ndarray]
        list of homographies to center image ([number_of_images x 3 x 3])
    T : np.ndarray
        translation matrix for panorama ([3 x 3])

    Returns
    ---------
    np.ndarray
        (result) blended panorama image ([height x width x 3])
    """
    
    # student_code start

    # Create alpha masks
    alpha_masks = []
    trans_images = []
    for i in range(len(images)):
        # Image
        trans_img = cv2.warpPerspective(images[i], T.dot(H[i]), (width, height))
        trans_images.append(trans_img)
        # Alpha mask
        h, w, _ = images[i].shape
        ones_mask = np.ones((h-2, w-2))
        mask = cv2.copyMakeBorder(ones_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        mask = ndimage.distance_transform_edt(mask)
        trans_mask = cv2.warpPerspective(mask, T.dot(H[i]), (width, height))
        alpha_masks.append(trans_mask)

    sum_alpha = sum(alpha_masks)
    alpha_masks = [alpha/sum_alpha for alpha in alpha_masks]
    alpha_masks = [cv2.merge((alpha, alpha, alpha)) for alpha in alpha_masks]
    weighted_images = [trans_images[i]*alpha_masks[i] for i in range(len(images))]

    result = np.zeros((height, width, 3))
    for i in range(len(images)):
        result = cv2.addWeighted(result, 1, weighted_images[i], 1, 0)
    result[np.isnan(result)] = 0
    result = result/255

    # student_code end

    return result, alpha_masks
