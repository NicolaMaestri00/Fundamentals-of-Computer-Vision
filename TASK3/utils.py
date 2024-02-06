#####TUWIEN - WS2023 CV: Task3 - Scene recognition with Bag of Visual Words
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_confusion_matrix(gt_labels, predicted, accuracy, class_names, group_no=None, name=None):
    """
    Plots a confusion matrix with given ground truth data 'gt_labels' and 'predicted' labels
    The parameter 'accuracy' should be the overal accuracy of the tested model
    gt_labels : ground truth of the test dataset [num_images x 1] - int
    predicted : predicted labels [num_images x 1] - int
    accuracy : overall accuracy (score) of the model - float
    class_names : list of class names [num_of_classes x 1] - string
    group_no : your group number (optional)
    name : filename (without extension) (optional)
    """

    labels = []
    labels.extend(gt_labels)
    labels.extend(predicted)
    labels = list(set(labels))
    labels.sort()
    class_names = [class_names[i] for i in labels]

    cm = confusion_matrix(gt_labels, predicted)

    fig, ax = plt.subplots()

    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    labels = np.arange(len(class_names))
    plt.xticks(labels, class_names, rotation=45)
    plt.yticks(labels, class_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    ax.set_title("test accuracy=" + str(accuracy))
    if name is not None and group_no is not None:
        ax.set_title(str(group_no) + ": test accuracy=" + str(accuracy))
        plt.savefig(os.path.join(os.getcwd(), 'results', name))

    plt.show()
