# **Predicting Schizophrenia Using Brain Anatomy**

This project was carried out as part of an exam-project of Professor Edouard Duchesnay's Machine Learning material in the MoSEF Data Science Course of Paris 1 Panthéon Sorbonne. 

This project aims to classify individuals based on whether they have schizophrenia or not, using features extracted from brain anatomy. Schizophrenia is associated with complex and diffuse brain atrophy, and this work focuses on predicting the clinical status of patients (schizophrenia vs. healthy control) using grey matter measurements in the brain.

The developed model classifies individuals based on the presence or absence of the disease, which could have important implications for early diagnosis of schizophrenia.

## **Dataset**

### **Description**
We have two types of data:

- **Training Set**: 410 samples
- **Test Set**: 103 samples

The data includes two types of features:

1. **Regions of Interest (ROIs)**: Information on the grey matter of the brain (284 features).
2. **VBM 3D Maps**: Three-dimensional representations of the brain's grey matter in MNI space (331,695 features).

### **Target**
The target is a binary variable distinguishing healthy controls from schizophrenia patients.

## **Feature Selection**

### **Dimensionality Reduction**
Given the large number of features (over 330,000 variables), we decided to reduce the dimensionality to avoid overfitting and speed up model training. The first step is to keep only the ROIs, as their information is redundant with that of the VBM maps. Next, we perform a correlation analysis of the ROIs to remove highly correlated variables.

- For continuous variables, we analyzed normality using the Shapiro-Wilk, Kolmogorov-Smirnov, and Jarque-Bera tests. Variables were considered normal if all three tests agreed.
- Pearson's correlation is used for normal variables, with a threshold of 0.9 to detect redundancy.
- For non-normal continuous variables, we used Spearman's test and applied the same correlation threshold.

After this analysis, we reduced the number of features from 280 to 214, which is a significant reduction. Logistic regression was then used to select the most impactful features, reducing the number of features to 82.

## **Models**

### **Models Tested**
We explored several non-linear models from different families of classification algorithms, including:

- **Support Vector Machine Classifier (SVC)**
- **Random Forest**
- **Gradient Boosting**
- **Multi-Layer Perceptron (MLP)**

### **Model Combination**
The final model is based on a combination of the following models:

- **Gradient Boosting**: This model captures complex relationships between features through an ensemble of decision trees.
- **Multi-Layer Perceptron (MLP)**: MLP can model complex, non-linear relationships and also extract advanced representations of the data, which is particularly useful given the high number of features.
- **Support Vector Classifier (SVC)**: The SVC is used to stabilize final predictions and help separate the schizophrenia and control classes.

We chose a **Voting Classifier** to combine these models.

### **Results**
The final model achieved an AUC score of 0.85 and a Balanced Accuracy of 0.76. We tested our models with different sets of features (1000 features, the initial 284 features, and the 82 selected features) and observed the impact of feature selection on model performance. Including all features increases the risk of overfitting, and the model performs better on the training set and worse on the test set when all features are used, indicating that the model is overfitting and struggles to generalize.


## Submission (Run locally)

After cloning the repo, you should follow these instructions :

### Installation

This starting kit requires Python and the following dependencies:

numpy
scipy
pandas
scikit-learn
matplolib
seaborn
jupyter
ramp-workflow

To run a submission and the notebook you will need the dependencies listed in requirements.txt. You can install the dependencies with the following command-line:

pip install -U -r requirements.txt

### Getting started

1. download the data locally:
python download_data.py
2. Running the model locally:
ramp-test --submission starting_kit

