import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

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


def plot_results(test_data, test_labels, predicted_classes, probs, classifier):
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

        plt.show()

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
    plt.savefig("OUTPUT/probs_and_correctness.png")

    # Plot confusion matrix
    plt.figure()
    plot_confusion_matrix(classifier.stats["confusion_matrix"], n_categories)
    plt.savefig("OUTPUT/confusion_matrix.png")