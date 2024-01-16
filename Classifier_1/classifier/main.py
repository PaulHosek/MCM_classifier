import matplotlib.pyplot as plt
import numpy as np
from src.loaders import load_data, load_labels
from src.classify import MCM_Classifier
from src.plot import  plot_results
import os
from src.plot import plot_confusion_matrix, plot_label_prob_diff

# Customizable environment variables
n_categories = 10  # Number of categories to be classified
n_variables = 121  # Number of variables in the dataset
mcm_filename_format = "train-images-unlabeled-{}_comms.dat"
data_filename_format = "train-images-unlabeled-{}.dat"

def main():
    print("{:-^50}".format("  MCM-Classifier  ")) 

    test_data = load_data("INPUT/data/test-images-unlabeled-all-uniform.txt").astype(int)
    test_labels = load_labels("INPUT/data/test-labels-uniform.txt").astype(int)

    # Step 1: Initialize classifier
    classifier = MCM_Classifier(
        n_categories, n_variables, mcm_filename_format, data_filename_format
    )

    # Step 2: Train
    classifier.fit(greedy=True, max_iter=1000000, max_no_improvement=100000)


    # Step 3: Evaluate
    predicted_classes, probs = classifier.evaluate(test_data, test_labels)
    np.savetxt("probs.csv", probs)

    # Step 4: Save classification report and other stats
    # report = classifier.get_classification_report(test_labels)
    classifier.save_classification_report(test_labels)

    if (classifier.stats == None):
        raise Exception("Classifier stats not found. Did you forget to call evaluate()?")

    # Count amount of -1 labels
    plot_results(test_data, test_labels, predicted_classes, probs, classifier)


# test for git



if __name__ == "__main__":
    main()
