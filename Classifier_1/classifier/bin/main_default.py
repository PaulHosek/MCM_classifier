import sys
sys.path.append("../")

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
data_path = "../INPUT/data/"
communities_path = "../INPUT/MCMs/" # FIXME THIS IS WEIRD. THIS SHOULD BE IN OUTPUT,
                                    # or wherever the SAA it writes it to.
                                    # In INPUT should only be those that are used as basis.
output_path = "../OUTPUT/"

def main():
    print("{:-^50}".format("  MCM-Classifier  ")) 

    test_data = load_data("../INPUT/data/test-images-unlabeled-all-uniform.txt").astype(int) # TODO this is still hardcoded
    test_labels = load_labels("../INPUT/data/test-labels-uniform.txt").astype(int)

    # Step 1: Initialize classifier
    classifier = MCM_Classifier(n_categories, n_variables, mcm_filename_format, data_filename_format, data_path, communities_path)

    # Step 2: Train
    classifier.fit(greedy=True, max_iter=1000000, max_no_improvement=100000)


    # Step 3: Evaluate
    predicted_classes, probs = classifier.predict(test_data, test_labels)

    # Step 4: Save classification report and other stats
    # report = classifier.get_classification_report(test_labels)
    classifier.save_classification_report(test_labels,path=output_path)

    if (classifier.stats == None):
        raise Exception("Classifier stats not found. Did you forget to call predict()?")

    # Count amount of -1 labels
    plot_results(test_data, test_labels, predicted_classes, probs, classifier,output_path,n_categories)





if __name__ == "__main__":
    main()
