import os.path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np






def draw_border(x, y, side, ax=None, **kwargs):
    """
    Draw a border on a specific side of a rectangle (or cell in imshow).

    :param x: The x-coordinate of the rectangle.
    :type x: float
    :param y: The y-coordinate of the rectangle.
    :type y: float
    :param side: The side of the rectangle to draw the border on. Valid values are 't', 'r', 'b', 'l'.
    :type side: str
    :param ax: The matplotlib axes object to draw on, defaults to None.
    :type ax: matplotlib.axes.Axes, optional
    :raises ValueError: If an invalid side value is provided.
    :return: The matplotlib Rectangle object representing the border.
    :rtype: matplotlib.patches.Rectangle
    """
    side_to_offset = {'t': (-.5, -.5, 1, 0), 'r': (.5, -.5, 0, 1), 'b': (-.5, .5, 1, 0), 'l': (-.5, -.5, 0, 1)}

    if side not in side_to_offset:
        raise ValueError("Invalid side value. Valid values are 't', 'r', 'b', 'l'.")
    
    dx, dy, width, height = side_to_offset[side]
    rect = plt.Rectangle((x+dx, y+dy), width, height, fill=False, **kwargs)
    
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect


def find_borders(arr):
    """
    Find the borders of different regions in a 2D array.

    :param arr: The input 2D array.
    :type arr: numpy.ndarray
    :return: A 2D list representing the borders of different regions.
             Each element in the list is a list of border directions ('t', 'b', 'l', 'r').
    :rtype: list[list[str]]
    """
    rows, cols = arr.shape
    borders = [[[] for _ in range(cols)] for _ in range(rows)]
    
    for i in range(rows):
        for j in range(cols):
            if i > 0 and arr[i, j] != arr[i-1, j]:  # Check top
                borders[i][j].append('t')

            if i < rows-1 and arr[i, j] != arr[i+1, j]:  # Check bottom
                borders[i][j].append('b')
            if j > 0 and arr[i, j] != arr[i, j-1]:  # Check left
                borders[i][j].append('l')
            if j < cols-1 and arr[i, j] != arr[i, j+1]:  # Check right
                borders[i][j].append('r')
    return borders












































































# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
#                        CLASSIFIER EVAL CODE PEPIJN
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------

def plot_confusion_matrix(confusion_matrix, n_categories: int, title="Confusion matrix", cmap="Blues", logScale: bool = False):
    """
    This function prints and plots the confusion matrix.
    
    Args:
        confusion_matrix (np.ndarray): Confusion matrix
        n_categories (int): Number of categories
        title (str, optional): Title of the plot. Defaults to "Confusion matrix".
        cmap (str, optional): Color map. Defaults to "Blues".
    """
    if logScale:
        plt.matshow(confusion_matrix, interpolation="nearest", cmap=cmap, norm=LogNorm(vmin=1, vmax=confusion_matrix.max()))
    else:
        plt.matshow(confusion_matrix, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(n_categories)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    
def plot_label_prob_diff(label1, label2, test_labels, probs, predicted_classes, title="Label probability difference"):
    correctly_classified_as_label1 = []
    correctly_classified_as_label2 = []
    incorrectly_classified_as_label1 = []
    incorrectly_classified_as_label2 = []
    for i in range(len(probs)):
        # Correctly classified as category 'label1'
        if test_labels[i] == label1 and predicted_classes[i] == label1:
            correctly_classified_as_label1.append(probs[i])
        # Correctly classified as category 'label2'
        if test_labels[i] == label2 and predicted_classes[i] == label2:
            correctly_classified_as_label2.append(probs[i])
        # Incorrectly classified as category 'label1'
        if test_labels[i] == label2 and predicted_classes[i] == label1:
            incorrectly_classified_as_label1.append(probs[i])
        # Incorrectly classified as category 'label2'
        if test_labels[i] == label1 and predicted_classes[i] == label2:
            incorrectly_classified_as_label2.append(probs[i])
            
    # Plot probabilities and correctness for categories 0 and 1
    plt.figure()
    # Correctly classified as category 3
    plt.scatter(
        np.array(correctly_classified_as_label1)[:, 3],
        np.array(correctly_classified_as_label1)[:, 5],
        color="green",
        marker="o", # type: ignore
        alpha=0.5,
        label=f"Correctly classified as {label1}",
    )
    # Correctly classified as category 5
    plt.scatter(
        np.array(correctly_classified_as_label2)[:, 3],
        np.array(correctly_classified_as_label2)[:, 5],
        color="green",
        marker="^", # type: ignore
        alpha=0.5,
        label=f"Correctly classified as {label2}",
    )
    # Incorrectly classified as category 3
    plt.scatter(
        np.array(incorrectly_classified_as_label1)[:, 3],
        np.array(incorrectly_classified_as_label1)[:, 5],
        color="red",
        marker="o", # type: ignore
        alpha=0.5,
        label=f"Incorrectly classified as {label1}",
    )
    # Incorrectly classified as category 5
    plt.scatter(
        np.array(incorrectly_classified_as_label2)[:, 3],
        np.array(incorrectly_classified_as_label2)[:, 5],
        color="red",
        marker="^", # type: ignore
        alpha=0.5,
        label=f"Incorrectly classified as {label2}",
    )
    plt.plot([0, 1], [0, 1], color="black", label="Perfect classifier")
    plt.plot([1, 0], [1, 0], color="black")
    plt.title(title)
    plt.xlabel(f"Probability of category {label1}")
    plt.ylabel(f"Probability of category {label2}")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()


def plot_results(test_data, test_labels, predicted_classes, probs, classifier, output_path,n_categories):
    # Count amount of -1 labels
    n_no_labels = 0
    no_labels_labels = []
    images = []
    for i in range(len(test_labels)):
        if predicted_classes[i] == -1:
            n_no_labels += 1
            no_labels_labels.append(test_labels[i])
            images.append(test_data[i])
    if n_no_labels != 0:
        # Plot all images with no labels so that they are in the same figure in a square grid
        dim_1 = int(np.sqrt(n_no_labels))
        dim_2 = int(np.sqrt(n_no_labels))
        fig, axs = plt.subplots(dim_1, dim_2, figsize=(10, 10))
        for i in range(dim_1):
            for j in range(dim_2):
                axs[i, j].imshow(images[i * dim_2 + j].reshape(11, 11), cmap="gray")
                axs[i, j].set_title(no_labels_labels[i * dim_2 + j])
                axs[i, j].axis("off")

        plt.savefig(os.path.join(output_path, "test.png"))

    print("Number of datapoints for which the classifier didn't have any probability for any category: {}".format(
        n_no_labels))
    print("Labels of these datapoints: {}".format(no_labels_labels))

    # Find amount of datapoints with 2 or more categories with probability > 0
    n_multiple_probs = 0
    multiple_probs_labels = []
    for i in range(len(probs)):
        if np.sum(probs[i] > 0) > 1:
            n_multiple_probs += 1
            multiple_probs_labels.append(test_labels[i])
    print("Number of datapoints with 2 or more categories with probability > 0: {}".format(n_multiple_probs))

    plot_label_prob_diff(3, 5, test_labels, probs, predicted_classes)
    plt.savefig(os.path.join(output_path, "probs_and_correctness.png"))

    # Plot confusion matrix
    plt.figure()
    plot_confusion_matrix(classifier.stats["confusion_matrix"], n_categories)
    plt.savefig(os.path.join(output_path, "confusion_matrix.png"))