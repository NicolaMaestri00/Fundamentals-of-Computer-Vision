#####TUWIEN - WS2023 CV: Task2 - Image Stitching
#####*********+++++++++*******++++INSERT GROUP NO. 4
from typing import List, Tuple
from numpy.linalg import inv
import numpy as np
import mapping
import random
import cv2


def get_geometric_transform(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Calculate a homography from the first set of points (p1) to the second (p2)

    Parameters
    ----------
    p1 : np.ndarray
        first set of points
    p2 : np.ndarray
        second set of points
    
    Returns
    ----------
    np.ndarray
        homography from p1 to p2
    """

    num_points = len(p1)
    A = np.zeros((2 * num_points, 9))
    for p in range(num_points):
        first = np.array([p1[p, 0], p1[p, 1], 1])
        A[2 * p] = np.concatenate(([0, 0, 0], -first, p2[p, 1] * first))
        A[2 * p + 1] = np.concatenate((first, [0, 0, 0], -p2[p, 0] * first))
    U, D, V = np.linalg.svd(A)
    H = V[8].reshape(3, 3)

    # homography from p1 to p2
    return (H / H[-1, -1]).astype(np.float32)


def get_transform(kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint], matches: List[cv2.DMatch]) -> Tuple[np.ndarray, List[int]]:
    """
    Estimate the homography between two set of keypoints by implementing the RANSAC algorithm
    HINT: random.sample(..), transforms.get_geometric_transform(..), cv2.perspectiveTransform(..)

    Parameters
    ----------
    kp1 : List[cv2.KeyPoint]
        keypoints left image ([number_of_keypoints] - KeyPoint)
    kp2 :  List[cv2.KeyPoint]
        keypoints right image ([number_of_keypoints] - KeyPoint)
    matches : List[cv2.DMatch]
        indices of matching keypoints ([number_of_matches] - DMatch)
    
    Returns
    ----------
    np.ndarray
        homographies from left (kp1) to right (kp2) image ([3 x 3] - float)
    List[int]
        inliers : list of indices, inliers in 'matches' ([number_of_inliers x 1] - int)
    """

    # student_code start

    N = 1000
    T = 5
    sample_size = 4
    H_max = np.zeros((3, 3))
    max_inliers = 0
    inliers = []

    for i in range(N):
        # Random Sample
        matches_sample = random.sample(matches, sample_size)

        # Get Homography
        list_kp1 = [kp1[mat.queryIdx].pt for mat in matches_sample] 
        list_kp2 = [kp2[mat.trainIdx].pt for mat in matches_sample]
        
        H = get_geometric_transform(np.array(list_kp1), np.array(list_kp2))

        # Get Inliers
        actual_inliers = []
        list_kp1 = [kp1[mat.queryIdx].pt for mat in matches]
        list_kp2 = [kp2[mat.trainIdx].pt for mat in matches]
        src = np.array(list_kp1).reshape(-1, 1, 2)
        dst = np.array(list_kp2).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(src, H)

        for j in range(len(dst)):
            dist = np.sqrt(np.sum((dst[j] - out[j]) ** 2))
            if dist < T:
                actual_inliers.append(matches[j])
                if matches[j] not in inliers and matches[j] not in matches_sample:
                    inliers.append(matches[j])

        if len(actual_inliers) > max_inliers:
            H_max = H
            max_inliers = len(actual_inliers)
    
    # Reestimate Homography based only on iliers
    list_kp1 = [kp1[mat.queryIdx].pt for mat in inliers] 
    list_kp2 = [kp2[mat.trainIdx].pt for mat in inliers]
    H = get_geometric_transform(np.array(list_kp1), np.array(list_kp2))

    # Check if this reestimate is better than the one before
    list_kp1 = [kp1[mat.queryIdx].pt for mat in matches]
    list_kp2 = [kp2[mat.trainIdx].pt for mat in matches]
    src = np.array(list_kp1).reshape(-1, 1, 2)
    dst = np.array(list_kp2).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(src, H)
    
    counter = 0
    for j in range(len(dst)):
        dist = np.sqrt(np.sum((dst[j] - out[j]) ** 2))
        if dist < T:
            counter += 1
            actual_inliers.append(matches[j])
    
    if counter > max_inliers:
        H_max = H

    # print("Number of Outliers: ", len(matches)-len(inliers))

    # student_code end

    return H_max, inliers


def to_center(desc: List[np.ndarray], kp: List[cv2.KeyPoint]) -> List[np.ndarray]:
    """
    Prepare all homographies by calculating the transforms from all other images
    to the reference image of the panorama (center image)
    First use mapping.calculate_matches(..) and get_transform(..) to get homographies between
    two consecutive images from left to right, then calculate and return the homographies to the center image
    HINT: inv(..), pay attention to the matrix multiplication order!!
    
    Parameters
    ----------
    desc : List[np.ndarray]
        list of descriptors ([number_of_images x num_of_keypoints, 128] - float)
    kp : List[cv2.KeyPoint]
        list of keypoints ([number_of_images x number_of_keypoints] - KeyPoint)
    
    Returns
    ----------
    List[np.ndarray]
        (H_center) list of homographies to the center image ( [number_of_images x 3 x 3] - float)
    """

    # student_code start

    H = []
    for i in range(len(desc)-1):
        list_matches = mapping.calculate_matches(desc[i], desc[i+1])
        H_b,_ = get_transform(kp[i], kp[i+1], list_matches)
        H.append(H_b)

    H_center = []
    a = int(len(H)/2)
    for i in range(len(H)+1):
        H_c = np.eye(3)
        if i < a:
            for j in range(i, a):
                H_c = np.matmul(H[j], H_c)
        elif i == a:
            H_c = np.eye(3)
        else:
            for j in range(i-1, a-1, -1):
                H_c = np.matmul(inv(H[j]), H_c)
        H_center.append(H_c)
    
    # student_code end

    return H_center


def get_panorama_extents(images: List[np.ndarray], H: List[np.ndarray]) -> Tuple[np.ndarray, int, int]:
    """
    Calculate the extent of the panorama by transforming the corners of every image
    and geht the minimum and maxima in x and y direction, as you read in the assignment description.
    Together with the panorama dimensions, return a translation matrix 'T' which transfers the
    panorama in a positive coordinate system. Remember that the origin of opencv images is in the upper left corner
    HINT: cv2.perspectiveTransform(..)

    Parameters
    ----------
    images : List[np.ndarray]
        list of images
    H : List[np.ndarray]
        list of homographies to center image ([number_of_images x 3 x 3])

    Returns
    ---------
    np.ndarray
        T : transformation matrix to translate the panorama to positive coordinates ([3 x 3])
    int
        width of panorama (in pixel)
    int
        height of panorama (in pixel)
    """

    # student_code start

    # Get Corners in the transformed image
    Corners_list = []
    for i in range(len(images)):
        h, w = images[0].shape[:2]
        cors = [[0, 0], [w, 0], [w, h], [0, h]]
        src = np.array(cors, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(src, H[i])
        Corners_list.append(out.reshape(4, 2))

    Corners_array = np.concatenate(Corners_list)
    [x_max, y_max] = np.max(Corners_array, axis=0)
    [x_min, y_min] = np.min(Corners_array, axis=0)

    # Calculate width and height
    width = int(x_max-x_min)
    height = int(y_max-y_min)

    # Translation Matrix
    T = np.eye(3)
    T[0,2] = abs(x_min)
    T[1,2] = abs(y_min)

    # student_code end

    return T, width, height
