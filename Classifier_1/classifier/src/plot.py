import os.path
import matplotlib.pyplot as plt
import numpy as np
import colorsys
import numpy as npu
import matplotlib.colors as mcolors


## ----- Partition Map ----- ## 
def generate_p_icc(data, P_MCM, n_variables,MCM,icc_idx):
    """Get probability distribution of single icc for some set of input images.
    E.g., what probabilities do we get for the first ICC in the MCM for the image of a 5 
    if we show it data of a 3. 
    ==> How good does this ICC differentiate between classes?
    P_MCM: is the probability distribution for the selected MCM
    """
    
    icc_Ps = np.zeros(len(data))
    for k, img in enumerate(data):
        idx = [i for i in range(n_variables) if MCM[icc_idx][i] == "1"]
        sm = int("".join([str(s) for s in img[idx]]), 2)
        icc_Ps[k] = P_MCM[icc_idx][sm] 
    return icc_Ps


def create_white_cmap():
    """
    Create a white colormap.

    :return: The white colormap.
    :rtype: matplotlib.colors.LinearSegmentedColormap
    """
    white_cmap = mcolors.LinearSegmentedColormap.from_list("white_cmap", [(1, 1, 1), (1, 1, 1)])
    return white_cmap


def create_pastel_cmap(num_hues, saturation=.4, value=.8):
    """
    Create a pastel colormap.

    :param num_hues: The number of hues in the colormap.
    :type num_hues: int
    :param saturation: The saturation value in the HSV color space.
    :type saturation: float
    :param value: The value (brightness) value in the HSV color space.
    :type value: float
    :return: The pastel colormap.
    :rtype: matplotlib.colors.LinearSegmentedColormap
    """
    hues = np.linspace(0, 1, num_hues)
    colors = [colorsys.hsv_to_rgb(hue, saturation, value) for hue in hues]
    pastel_cmap = mcolors.LinearSegmentedColormap.from_list("pastel_cmap", colors)
    return pastel_cmap



 
def draw_all_borders(borders,ax=None, linewidth=2, color="black", **kwargs):
    """
    Draw borders for all elements in the border list.

    :param linewidth: The width of the border lines.
    :type linewidth: int
    :param color: The color of the border lines.
    :type color: str
    :param **kwargs: Additional keyword arguments to be passed to plt.Rectangle.
    """
    for i in range(len(borders)):
        for j in range(len(borders[i])):
            for side in borders[i][j]:
                draw_border(j, i, side,ax=ax,linewidth=linewidth,color="black", **kwargs) # TODO Need to do -1/2 linewidth offset the in the direction of the border

def draw_all_values(vals, ax=None, color="white", **kwargs):
    """
    Draw all values in a grid.

    :param vals: The values to be drawn.
    :type vals: numpy.ndarray
    :param ax: The axes object to draw on, defaults to None.
    :type ax: matplotlib.axes.Axes, optional
    :param color: The color of the text, defaults to "white".
    :type color: str, optional
    """
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax = ax or plt.gca()
            txt = plt.text(j, i, vals[i, j], ha='center', va='center', color=color, **kwargs)


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
        plt.matshow(confusion_matrix, interpolation="nearest", cmap=cmap, norm=mcolors.LogNorm(vmin=1, vmax=confusion_matrix.max()))
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