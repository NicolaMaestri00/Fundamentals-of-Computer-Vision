#####TUWIEN - WS2023 CV: Task2 - Image Stitching
#####*********+++++++++*******++++INSERT GROUP NO. 4
from typing import List, Tuple
from matplotlib.pyplot import angle_spectrum
import numpy as np
# from cyvlfeat.sift.sift import sift
# import cyvlfeat
import os
import cv2
import glob


def get_panorama_data(path: str) -> Tuple[List[np.ndarray], List[cv2.KeyPoint], List[np.ndarray]]:
    """
    Loop through images in given folder (do not forget to sort them), extract SIFT points
    and return images, keypoints and descriptors
    This time we need to work with color images. Since OpenCV uses BGR you need to swap the channels to RGB
    HINT: sorted(..), glob.glob(..), cv2.imread(..), sift=cv2.SIFT_create(), sift.detectAndCompute(..)

    Parameters
    ----------
    path : str
        path to image folder

    Returns
    ---------
    List[np.ndarray]
        img_data : list of images
    List[cv2.Keypoint]
        all_keypoints : list of keypoints ([number_of_images x number_of_keypoints] - KeyPoint)
    List[np.ndarray]
        all_descriptors : list of descriptors ([number_of_images x num_of_keypoints, 128] - float)
    """
    
    # student_code start

    img_data = []
    all_keypoints = []
    all_descriptors = []

    list_img = sorted(os.listdir(path))

    for filename in list_img:
        img = cv2.imread(os.path.join(path,filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.cvtColor(cv2.imread(os.path.join(path,filename)), cv2.COLOR_BGR2RGB)
        if img is not None:
            img_data.append(img)
    
    for img in img_data:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img_gray, None)
        all_keypoints.append(keypoints)
        all_descriptors.append(descriptors)
    
    # student_code end

    return img_data, all_keypoints, all_descriptors
