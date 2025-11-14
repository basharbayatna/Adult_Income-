# Adult Income Classification Project

## 🎯 Project Overview
This project builds a machine learning model to predict whether an individual's income exceeds **$50,000 annually** using the Adult Income dataset (derived from the 1994 Census database).

### Core Objectives
- Exploratory Data Analysis (EDA), cleaning, and preprocessing  
- Handle class imbalance using **SMOTE**  
- Build and compare models: **Random Forest** and **Neural Networks (Keras)**  
- Apply **PCA** and **KMeans Clustering** for feature engineering  
- Tune Neural Network hyperparameters using **Keras Tuner**

---

## 💾 Dataset Details
The dataset contains demographic and employment-related features.

### Key Features
- `age` — Age of the individual  
- `workclass` — Type of employer  
- `educational-num` — Numerical education level  
- `marital-status` — Marital status  
- `occupation` — Occupation  
- `relationship` — Relationship status  
- `race` — Race  
- `gender` — Gender  
- `capital-gain`, `capital-loss` — Financial information  
- `hours-per-week` — Weekly working hours  
- `native-country` — Country of origin  

### Target Variable
- `income`: **HIGH** (> 50K) or **LOW** (≤ 50K)  
  *(Original values were mapped to LOW and HIGH during preprocessing.)*

---

## 🛠️ Methodology & Workflow

### 1. Data Preprocessing
- Replaced missing values (`?`) with `"MISSING"` or imputed constants  
- Used `ColumnTransformer` for preprocessing:
  - Numerical → `StandardScaler`
  - Categorical → imputation + `OneHotEncoder`
- Encoded target variable using `LabelEncoder`  
- Applied **SMOTE** to balance HIGH vs. LOW income classes  

### 2. Feature Engineering & Selection
- Applied **PCA** to extract top 3 components  
- Used **KMeans (k=3)** clustering and added cluster label as a new feature  
- Computed **Permutation Importance** to assess feature contributions, including engineered features  

### 3. Model Development
#### Random Forest Classifier
- Trained on SMOTE-resampled and fully processed data  

#### Neural Networks (Keras)
- **Baseline NN:** simple sequential model with one hidden layer  
- **Tuned NN:** hyperparameters optimized using **Keras Tuner (RandomSearch)** (units, dropout, optimizer, learning rate)

---

## 📊 Model Performance Comparison

| Metric       | Baseline NN | Tuned NN (Best) | Random Forest |
|--------------|-------------|------------------|---------------|
| Accuracy     | 0.8148      | 0.8407           | 0.8354        |
| Precision    | 0.6558      | 0.7303           | 0.7101        |
| Recall       | 0.6134      | 0.6385           | 0.6561        |
| F1-Score     | 0.6339      | 0.6811           | 0.6820        |

---

## 📈 Conclusion
The **Tuned Neural Network** achieved the highest Accuracy and F1-Score, making it the best-performing model for this task. Strong performance resulted from preprocessing, PCA + clustering, and hyperparameter tuning.

### Top 10 Most Important Features (Permutation Importance)
1. `marital-status_Married-civ-spouse`  
2. `educational-num`  
3. `capital-gain`  
4. `relationship_Husband`  
5. `age`  
6. `hours-per-week`  
7. `PCA_1` *(engineered feature)*  
8. `capital-loss`  
9. `relationship_Wife`  
10. `Cluster` *(engineered feature)*  

*Engineered features like PCA components and Cluster label proved highly predictive.*

---

## 🧾 What’s Included
- Preprocessing pipeline (ColumnTransformer)  
- SMOTE resampling  
- PCA & KMeans engineered features  
- Random Forest and Neural Network models  
- Keras Tuner hyperparameter optimization  
- Permutation importance feature analysis  

---

## 📎 Optional Additions
If needed, I can add:
- Installation & usage steps  
- Centered banner image  
- Result visuals (confusion matrix, ROC, feature plots)  
- Code snippets for inference  
