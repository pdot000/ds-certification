# 📊 Recipe Site Traffic Prediction - Data Science Project

## 🏆 Project Overview
This project is part of my **Data Scientist Certification**, where I was tasked with developing a model to **predict which recipes will lead to high website traffic**. The challenge was given by the **product team**.

The goal was to **correctly predict high-traffic recipes at least 80% of the time** while minimizing the risk of displaying unpopular ones. More traffic means more subscriptions, making this a key business problem.

## 👨‍💻 My Role
As the **Data Scientist**, I:
- **Cleaned and validated** the dataset for inconsistencies.
- **Explored the data** to identify trends and key drivers of recipe popularity.
- **Developed machine learning models** to predict high-traffic recipes.
- **Evaluated performance** using precision-recall metrics.
- **Provided business insights** based on the findings.

## 🛠️ Tools & Libraries Used

### Data Processing & Manipulation
- Pandas → Data handling, filtering, and transformation
- NumPy → Numerical computations and array operations
- Scikit-learn (ColumnTransformer, StandardScaler, OneHotEncoder) → Data preprocessing

### Exploratory Data Analysis (EDA) & Statistical Testing
- Matplotlib & Seaborn → Data visualization (histograms, boxplots, correlation heatmaps)
- Scipy (chi2_contingency) → Chi-squared test for categorical variable relationships
- Correlation Analysis → Identifying relationships between features
- Feature Distributions → Understanding variable behavior

### Machine Learning & Model Training
- Scikit-learn (GradientBoostingClassifier, SVC, LogisticRegression, etc.) → Supervised learning classification models
- Scikit-learn (RandomizedSearchCV) → Hyperparameter tuning
- Pipeline with ColumnTransformer → Streamlining preprocessing and model training

### Model Evaluation & Performance Metrics
- Creation of custom scorer (Precision at Fixed Recall - 80%) → Ensuring high-traffic recipes are prioritized
- Scikit-learn (roc_auc_score, f1_score, precision_recall_curve) → Performance evaluation

###  Model Explainability & Business Insights
- SHAP (SHapley Additive Explanations) → Feature impact visualization
- Support Vector Analysis → Identifying most influential data points

## 🛠️ Approach & Methodology

### **1️⃣ Data Validation & Cleaning**
- Checked for missing values, duplicates, and incorrect data types.
- Verified that the dataset matched the business problem requirements.
- Encoded categorical features and scaled numerical values.

### **2️⃣ Exploratory Data Analysis (EDA)**
- Identified trends in **calories, carbohydrates, sugar, protein, and servings**.
- Visualized **how different categories impact traffic**.
- Used **correlation heatmaps and distribution plots** to find patterns.

### **3️⃣ Model Development and Evaluation**
- **Baseline Model:** A simple model to establish initial performance.
- Other **Machine Learning Models** (Classifiers) e.g.:
    - Random Forest Classifier
    - Gradient Boosting Classifier
    - **Support Vector Machine (SVC)** → chosen model as the best
    - Logistic Regression for comparison
- Used **Precision at 80% Recall** to ensure high-traffic recipes are prioritized.
- Evaluated models with **ROC-AUC, F1-score, and Precision-Recall AUC**.
- Tuned hyperparameters with **RandomizedSearchCV**.

### **4️⃣ Business Insights & Recommendations**
- Recipes with **higher protein and carbohydrate content tend to attract more traffic**.
- Certain recipe categories (e.g., **Potato**, **Vegetable**) are more likely to generate high traffic.
- **Serving size plays a role** — recipes with more servings tend to be more popular.
- **Sugar content is less relevant** to recipe popularity compared to other features.
- **A machine learning model can assist in homepage recipe selection** to improve traffic prediction accuracy.

## 🖼️ Visualizations & Reports
- **Data Cleaning & EDA:** Histograms, Boxplots, and Correlation pairplots.
- **Feature Importance:** SHAP values and Permutation Importance.
