#####TUWIEN - WS2023 CV: Task0 - Colorizing Images
import matplotlib.pyplot as plt
import numpy as np


def show_plot(img: np.ndarray, group_no: str = None, name: str = None):
    """
    Display the given image using Matplotlib. Optionally, save the image to a folder.

    Args:
        img (np.ndarray): The image to be displayed.
        group_no (str, optional): Your group number (if provided, it will be used as the figure title).
        name (str, optional): The name of the saved image file (if provided, the image will be saved in a 'results' folder).

    Example usage:
        show_plot(image_data, group_no="Group 1", name="result_image.png")
    """

    fig = plt.figure()
    plt.imshow(img)

    if name is not None and group_no is not None:
        fig.suptitle(group_no)
        plt.savefig('results/' + name)

    plt.plot()
