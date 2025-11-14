Adult Income Classification Project

🎯 Project Overview

This project focuses on building a machine learning model to predict whether an individual's income exceeds $50,000 annually, based on the Adult Income dataset (often derived from the 1994 Census database).

The core objectives were:

Exploratory Data Analysis (EDA), cleaning, and preprocessing the data.

Handling the class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).

Developing and comparing two main classification models: Random Forest Classifier and Neural Networks (Keras).

Exploring dimensionality reduction using PCA and KMeans Clustering for feature engineering.

Optimizing the Neural Network model using Keras Tuner.

💾 Dataset Details

The dataset contains various demographic and employment-related features.

Key Features Used:

age: Age of the individual.

workclass: Type of employer.

educational-num: Numerical equivalent of the education level.

marital-status: Marital status.

occupation: Occupation.

relationship: Relationship status.

race: Race.

gender: Gender.

capital-gain, capital-loss: Financial information.

hours-per-week: Working hours per week.

native-country: Country of origin.

Target Variable:

income: Binary outcome indicating whether income is HIGH (>50K) or LOW (<=50K). (Note: Original values were mapped to LOW and HIGH in the notebook).

🛠️ Methodology and Workflow

1. Data Preprocessing

Missing Values: Handled missing values (originally marked as ? in the dataset) by replacing them with MISSING or imputing them with a constant value during preprocessing.

Feature Engineering: Created a custom preprocessing pipeline using ColumnTransformer:

Numerical Features: Scaled using StandardScaler.

Categorical Features: Imputed and encoded using OneHotEncoder.

Target Encoding: The target variable (income) was encoded using LabelEncoder.

Imbalance Handling: The heavily imbalanced training data was balanced using SMOTE to generate synthetic samples for the minority class (HIGH income).

2. Feature Engineering & Selection

Dimensionality Reduction: Principal Component Analysis (PCA) was applied to the processed data to capture the top 3 components.

Clustering: KMeans Clustering (with k=3 clusters, determined by Elbow and Silhouette methods) was applied, and the resulting cluster label was added as a new feature.

Permutation Importance: Used to evaluate the contribution of features (including engineered features like PCA components and the Cluster label) in the final Random Forest model.

3. Model Development

A. Random Forest Classifier

A baseline Random Forest Classifier was trained on the SMOTE-resampled and processed data.

B. Neural Network (Keras)

Two Neural Network models were developed:

Baseline NN: A simple sequential model with one hidden layer.

Tuned NN: An advanced model where hyperparameters (units, dropout rate, optimizer, and learning rate) were optimized using Keras Tuner (RandomSearch) to maximize validation accuracy.

📊 Key Results and Model Comparison

The models were evaluated primarily on Accuracy, Precision, Recall, and F1-Score on the hold-out test set.

Metric

Baseline NN

Tuned NN (Best Model)

Random Forest

Accuracy

0.8148

0.8407

0.8354

Precision

0.6558

0.7303

0.7101

Recall

0.6134

0.6385

0.6561

F1-Score

0.6339

0.6811

0.6820

📈 Conclusion:

The Tuned Neural Network model achieved the highest overall Accuracy and F1-Score, making it the best performing model for this classification task. The high performance is attributed to hyperparameter tuning and effective preprocessing.

Most Important Features (from Permutation Importance):

marital-status_Married-civ-spouse

educational-num

capital-gain

relationship_Husband

age

hours-per-week

PCA_1 (Engineered Feature)

capital-loss

relationship_Wife

Cluster (Engineered Feature)

Note: The engineered features (PCA components and the Cluster label) demonstrated significant predictive power, ranking within the top 10 most important features.
