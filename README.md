# A classifier based on minimal description of binary data.

# MCM Classifier

## Overview
This repository contains the implementation of the **Minimally Complex Model (MCM) Classifier**, developed as part of the Master's thesis titled *"Pattern recognition in higher-order interaction systems using models of low information-theoretic complexity"* by Paul J.A. Hosek at the University of Amsterdam. The work was conducted under the supervision of Dr. Clélia M.C. de Mulatier at the Computational Soft Matter Lab, Institute of Theoretical Physics, and Computational Science Lab, Informatics Institute.

The MCM Classifier is a novel approach to pattern recognition in complex systems with higher-order interactions. It leverages Minimally Complex Models (MCMs) to extract robust, interpretable, and data-specific features from binary datasets, such as a modified MNIST dataset. These features are used for classification tasks, demonstrating high accuracy and stability even in undersampling regimes. The repository includes code for model fitting, feature extraction, and classification, along with supporting documentation.

## Features
- **Feature Extraction**: Extracts simple and interpretable features using MCMs, capturing higher-order interactions without parameter fitting.
- **Robustness**: Demonstrates stability in feature extraction under varying sample sizes, with convergence to reference models at approximately 1000 samples.
- **Classification**: Implements a naive Bayes classifier based on MCM features, achieving a mean test accuracy of 93.06% on the modified MNIST dataset.
- **Comparison with Pairwise Models**: Includes implementations of sparse fully-connected pairwise models for benchmarking against MCMs.
- **Visualization**: Provides tools for visualizing Independent Complete Components (ICCs) and pixel-wise log-evidence.



## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/PaulHosek/MCM_classifier.git
   cd MCM_classifier
   ```

2. **Set Up Python Environment**:
   Ensure Python 3.8+ is installed. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Dataset**:
   The modified MNIST dataset used in the thesis is not included due to size constraints. You can preprocess the standard MNIST dataset using the provided scripts in `data/preprocessing/` or contact the author for access to the modified dataset.

## Usage
1. **Prepare the Dataset**:
   Place the modified MNIST dataset in the `data/` directory or preprocess the standard MNIST dataset:
   ```bash
   python data/preprocessing/preprocess_mnist.py
   ```

2. **Train MCM Models**:
   Fit MCMs on the dataset to extract features:
   ```bash
   python src/mcm/fit_mcm.py --data data/modified_mnist --output models/mcm
   ```

3. **Run Classification**:
   Use the trained MCMs for classification:
   ```bash
   python src/classifier/naive_bayes.py --models models/mcm --test data/test_set
   ```

4. **Visualize Results**:
   Generate visualizations of ICCs and log-evidence:
   ```bash
   python src/visualization/plot_icc.py --model models/mcm/digit_0
   ```

Example scripts are provided in the `examples/` directory for end-to-end workflows.

##  Context
The MCM Classifier is built upon the theoretical framework of Minimally Complex Models, which partition variables into non-interacting groups of highly correlated variables (Independent Complete Components, ICCs). Unlike traditional pairwise models, MCMs include all possible interactions within each ICC, resulting in low information-theoretic complexity. The thesis demonstrates:
- **Feature Interpretability**: MCMs extract features that reveal locality in the data, such as pixel-wise community structures in MNIST digits.
- **Classification Performance**: Subsets of ICCs can distinguish digits, even those with similar statistical structures (e.g., digits 3 and 5).
- **Computational Efficiency**: MCM fitting is significantly faster (1.5 seconds for 6315 samples) compared to pairwise models (non-converging after 4 hours).

For detailed methodology, results, and discussions, refer to the thesis PDF in the `docs/` directory.

## Requirements
- Python 3.8+
- NumPy
- SciPy
- Pandas
- Matplotlib
- Scikit-learn
- (Optional) Jupyter for running example notebooks

Install dependencies using:
```bash
pip install -r requirements.txt
```


## Related work

https://arxiv.org/abs/2008.00520

https://www.mdpi.com/1099-4300/20/10/739
