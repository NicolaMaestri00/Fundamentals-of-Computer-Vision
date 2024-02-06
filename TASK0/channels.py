#####TUWIEN - WS2023 CV: Task0 - Colorizing Images
#####*********+++++++++*******++++INSERT MATRICULAR NO. HERE 12306354
import matplotlib.pyplot as plt
import numpy as np


def corr2d(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate the normalized cross-correlation (NCC) between two input images.
    
    Args:
        img1 (np.ndarray): The first input image (n1 x m1 x 1).
        img2 (np.ndarray): The second input image (n2 x m2 x 1).
    
    Returns:
        float: The normalized cross-correlation coefficient between the images.
    """

    # student_code start
    corr = 0
    corr = np.sum((img1-np.mean(img1))*(img2-np.mean(img2)))/(np.sum((img1-np.mean(img1))**2)*np.sum((img2-np.mean(img2))**2))**0.5

    # student_code end
    
    return corr


def align(imgR: np.ndarray, imgG: np.ndarray, imgB: np.ndarray) -> np.ndarray:
    """
    Align color channels using normalized cross-correlation to create a colorized image.
    HINT: np.roll(..)

    Args:
        imgR (np.ndarray): Image representing the red channel.
        imgG (np.ndarray): Image representing the green channel.
        imgB (np.ndarray): Image representing the blue channel.
    
    Returns:
        np.ndarray: The colorized image (n x m x 3) as a numpy array.
    """
    
    # student_code start
    max_corr = 0
    x_shift = 0
    y_shift = 0
    
    for i in range(-15,16):
    	for j in range(-15,16):
    		imgG_shifted = np.roll(imgG, (i, j), axis=(1, 0))
    		corr = corr2d(imgR, imgG_shifted)
    		if max_corr < corr:
    			max_corr = corr
    			x_shift = i
    			y_shift = j
    
    imgG_shifted = np.roll(imgG, (x_shift, y_shift), axis=(1, 0))

    max_corr = 0
    x_shift = 0
    y_shift = 0
        
    for i in range(-15,16):
    	for j in range(-15,16):
    		imgB_shifted = np.roll(imgB, (i, j), axis=(1, 0))
    		corr = corr2d(imgR, imgB_shifted)
    		if max_corr < corr:
    			max_corr = corr
    			x_shift = i
    			y_shift = j
    			
    imgB_shifted = np.roll(imgB, (x_shift, y_shift), axis=(1, 0))
    
    result = np.dstack((imgR, imgG_shifted, imgB_shifted))
    
    # student_code end
                
    return result
