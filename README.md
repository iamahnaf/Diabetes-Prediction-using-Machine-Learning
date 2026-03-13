# Diabetes Prediction using Machine Learning

## Overview
This repository contains a machine learning project aimed at predicting diabetes based on various health metrics. The models in this project (such as K-Nearest Neighbors and Logistic Regression) were built **from scratch** to demonstrate a deep understanding of the underlying algorithms.

## Repository Contents
- **`Project.ipynb`**: The main Jupyter Notebook. It includes the entire pipeline: importing libraries, loading the data, preprocessing (Label Encoding, Scaling), exploratory data analysis (EDA), model training, and performance evaluation.
- **`uiu100k.csv`**: The dataset containing 100,000 patient records. Features include age, gender, smoking history, BMI, and blood glucose level.

## Visualizations & Exploratory Data Analysis
As part of the analysis (the pictures outputted in the Jupyter Notebook run), several visualizations were generated to understand the data distribution and relationships:

### Age Distribution
A histogram showing the frequency of different age groups in the dataset.
![Age Distribution](images/age_distribution.png)

### BMI vs. Blood Glucose Level
A scatter plot illustrating the relationship between BMI, Blood Glucose, and the presence of diabetes.
![BMI vs Blood Glucose](images/bmi_vs_glucose.png)

### Feature Correlation Heatmap
A Seaborn heatmap displaying how different features are correlated with one another, helping identify the most significant predictors of diabetes.
![Feature Correlation Heatmap](images/correlation_heatmap.png)

### Model Performance (Confusion Matrix)
Visualizing the performance of the machine learning models.
![Model Performance Matrix](images/knn_confusion_matrix.png)

## Models Implemented
The following models were coded completely from scratch without fully relying on standard library implementations for their core algorithms:
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**

## Technologies Used
- **Python** base language
- **Pandas & NumPy** for data manipulation and mathematical operations
- **Matplotlib & Seaborn** for data visualization
- **Scikit-Learn** for data preprocessing (StandardScaler, LabelEncoder), train-test splitting, and evaluation metrics (Accuracy, F1-Score, ROC/AUC, etc.)
