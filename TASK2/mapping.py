#####TUWIEN - WS2023 CV: Task2 - Image Stitching
from typing import List
import numpy as np
import cv2


def calculate_matches(des1: np.ndarray, des2: np.ndarray) -> List[int]:
    """
    Search for matching SIFT descriptors in two consecutive images
    with k- nearest neighbour search, using k = 2
    and applying the LOWE ratio test to remove possible outliers

    Parameters
    ----------
    des1 : np.ndarray
        SIFT descriptors of left image ([num_of_desc x 128] - float)
    des2 : np.ndarray
        SIFT descriptors of right image, ([num_of_desc x 128]- float)

    Returns
    ---------
    List[int]
        (result) indices of keypoints (kp1 (col1) and kp2 (col2)) per match ([num_of_matches x 2] - int)
    """

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test - LOWE
    result = []
    for m, n in matches:
        if (m.distance / n.distance) < 0.8:
            result.append(m)

    return result
