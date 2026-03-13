# Diabetes Prediction using Machine Learning

## Overview
This repository contains a machine learning project aimed at predicting diabetes based on various health metrics. The models in this project (such as K-Nearest Neighbors and Logistic Regression) were built **from scratch** to demonstrate a deep understanding of the underlying algorithms.

## Repository Contents
- **`universityProject.ipynb`**: The main Jupyter Notebook. It includes the entire pipeline: importing libraries, loading the data, preprocessing (Label Encoding, Scaling), exploratory data analysis (EDA), model training, and performance evaluation.
- **`uiu100k.csv`**: The dataset containing 100,000 patient records. Features include age, gender, smoking history, BMI, and blood glucose level.

## Visualizations & Exploratory Data Analysis
As part of the analysis (the pictures outputted in the Jupyter Notebook run), several visualizations were generated to understand the data distribution and relationships:
- **Age Distribution:** A histogram showing the frequency of different age groups in the dataset.
- **BMI vs. Blood Glucose Level:** A scatter plot illustrating the relationship between BMI, Blood Glucose, and the presence of diabetes.
- **Feature Correlation Heatmap:** A Seaborn heatmap displaying how different features are correlated with one another, helping identify the most significant predictors of diabetes.

## Models Implemented
The following models were coded completely from scratch without fully relying on standard library implementations for their core algorithms:
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**

## Technologies Used
- **Python** base language
- **Pandas & NumPy** for data manipulation and mathematical operations
- **Matplotlib & Seaborn** for data visualization
- **Scikit-Learn** for data preprocessing (StandardScaler, LabelEncoder), train-test splitting, and evaluation metrics (Accuracy, F1-Score, ROC/AUC, etc.)

## How to Run
1. Clone the repository.
2. Ensure you have Jupyter Notebook or JupyterLab installed, along with the required libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`).
3. Open `universityProject.ipynb` and run the cells sequentially to see the data processing, visualizations, and model training in action.
