# TUWIEN - WS2023 CV: Task3 - Scene recognition using Bag of Visual Words
# *********+++++++++*******++++INSERT GROUP NO. 4
from typing import List
import sklearn.metrics.pairwise as sklearn_pairwise
import cv2
import numpy as np
import random
import time


def extract_dsift(images: List[np.ndarray], stepsize: int, num_samples: int = None) -> List[np.ndarray]:
    """
    Extracts dense feature points on a regular grid with 'stepsize' and optionally returns
    'num_samples' random samples per image. If 'num_samples' is not provided, it takes all
    features extracted with the given 'stepsize'. SIFT.compute has the argument "keypoints",
    which should be set to a list of keypoints for each square.
    
    Args:
    - images (List[np.ndarray]): List of images to extract dense SIFT features [num_of_images x n x m] - float
    - stepsize (int): Grid spacing, step size in x and y direction.
    - num_samples (int, optional): Random number of samples per image.

    Returns:
    - List[np.ndarray]: SIFT descriptors for each image [number_of_images x num_samples x 128] - float
    """
    tic = time.perf_counter()

    # student_code start
    all_descriptors = []
    sift = cv2.SIFT_create()

    for image in images:
        keypoints = [cv2.KeyPoint(x, y, stepsize) for y in range(0, image.shape[0], stepsize) for x in range(0, image.shape[1], stepsize)]
        if num_samples is not None:
            keypoints = random.sample(keypoints, num_samples)
        img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        _, descriptors = sift.compute(img, keypoints)
        all_descriptors.append(descriptors)

    # student_code end

    toc = time.perf_counter()
    print("DSIFT Extraction:", toc - tic, " seconds")

    # all_descriptors : list sift descriptors per image [number_of_images x num_samples x 128] - float
    return all_descriptors


def count_visual_words(dense_feat: List[np.ndarray], centroids: List[np.ndarray]) -> List[np.ndarray]:
    """
    For classification, generates a histogram of word occurrences per image.
    Utilizes sklearn_pairwise.pairwise_distances(..) to assign the descriptors per image
    to the nearest centroids and counts the occurrences of each centroid. The histogram
    should be as long as the vocabulary size (number of centroids).

    Args:
    - dense_feat (List[np.ndarray]): List of SIFT descriptors per image [number_of_images x num_samples x 128] - float
    - centroids (List[np.ndarray]): Centroids of clusters [vocabulary_size x 128]

    Returns:
    - List[np.ndarray]: List of histograms per image [number_of_images x vocabulary_size]
    """
    tic = time.perf_counter()

    # student_code start
    histograms = []

    for image in dense_feat:
        distances = sklearn_pairwise.pairwise_distances(image, centroids)
        histogram = np.zeros(centroids.shape[0])
        for distance in distances:
            histogram[np.argmin(distance)] += 1
        histograms.append(histogram)
    
    # student_code end

    toc = time.perf_counter()
    print("Counting visual words:", toc - tic, " seconds")

    # histograms : list of histograms per image [number_of_images x vocabulary_size]
    return histograms