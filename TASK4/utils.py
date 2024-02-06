# TUWIEN - WS2023 CV: Task4 - Mask Classification using CNN
# *********+++++++++*******++++INSERT GROUP NO. HERE
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import torch
import torch.nn as nn


def plot_activation_maps(model, image_sample, group_no=None, name=None):
    """
    Plots samples of activation maps of each convolutional layer of the given 'model'
    Uses the given 'image_sample' to obtain the activation maps.
    Use additional arguments to save the image to folder /results in your local directory with given 'name' and 'group_no'
    model : extract the activation maps and kernel from this model
    image_sample : single image sample [100 x 100 x 1]
    group_no : your group number (optional)
    name : filename (without extension) (optional)
    """

    filters = []  # we will save the conv layer weights in this list
    conv_layers = []  # we will save the 49 conv layers in this list
    # get all the model children as list
    model_children = list(model.children())
    # counter to keep count of the conv layers
    counter = 0
    # append all the conv layers and their respective weights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            filters.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                child = model_children[i][j]
                if type(child) == nn.Conv2d:
                    counter += 1
                    filters.append(child.weight)
                    conv_layers.append(child)
    print(f"Total convolutional layers: {counter}")

    # pass the image through all the layers
    activations = [conv_layers[0](torch.unsqueeze(image_sample, axis=0))]
    for i in range(1, len(conv_layers)):
        # pass the result from the last layer to the next layer
        activations.append(conv_layers[i](activations[-1]))

    col_size = 5
    row_size = len(conv_layers)*2+1
    fig, ax = plt.subplots(
        row_size, col_size, figsize=(col_size*2, row_size*2))

    ax[0][0].imshow(image_sample.permute(1, 2, 0).numpy(), cmap="gray")
    ax[0][0].axis('off')
    for i in range(1, col_size):
        ax[0][i].axis('off')

    for layer in range(0, len(conv_layers)):
        ax[1+layer*2][0].set_title(f'conv layer {layer}',
                                   loc='left', color='white', fontsize=16, x=0.00, y=0.8)
        activation = activations[layer]
        filter = filters[layer]
        for col in range(0, col_size):
            if col < len(filter):
                ax[1+layer*2][col].imshow(activation[0,
                                          col, :, :].detach().numpy(), cmap='gray')
                ax[1+layer*2+1][col].imshow(filter[col,
                                            0, :, :].detach().numpy(), cmap='gray')
            ax[1+layer*2][col].axis('off')
            ax[1+layer*2+1][col].axis('off')

    if name is not None and group_no is not None:
        fig.suptitle(str(group_no))
        plt.savefig(os.path.join(os.getcwd(), 'results', name))

    fig.tight_layout()
    plt.subplots_adjust(wspace=.05, hspace=.05)
    plt.axis('off')
    plt.plot()
