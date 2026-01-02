# Machine Learning Regression and Classification Mini-Projects on sckit.ipynb

This repository features practical examples of regression and classification tasks using Python’s scikit-learn package. Each notebook/script demonstrates real-world data handling, model building, evaluation, and visualization on classic datasets.

---

## Objectives

- Learn to implement common machine learning algorithms for both regression and classification.
- Develop model evaluation skills using metrics and visualizations.
- Understand basic hyperparameter tuning with GridSearchCV.
- Practice exploratory data analysis and feature importance interpretation.

---

## Datasets

- **Diabetes** (Regression)
- **Boston Housing** (Regression)
- **Breast Cancer Wisconsin** (Classification)

---

## Requirements

- Python 3.7+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- scipy


## Work Done

### 1. **Decision Tree Regression on Diabetes**

**Objective:**  
- Predict diabetes progression one year after baseline using patient attributes.
- Assess DecisionTreeRegressor on this regression task.

**What I Did:**  
- Loaded the diabetes dataset.
- Split into train and test sets.
- Trained a DecisionTreeRegressor.
- Evaluated using R² score, MAE, and MSE.
- Visualized prediction distributions and actual-vs-predicted.

**Results:**  
- Got model metrics (R², MAE, MSE).
- Distribution plot showed range of predicted values.
- Actual vs. predicted scatterplot gave insight into model accuracy and error spread.

---

### 2. **Linear Regression (Pipeline) on Boston Housing**

**Objective:**  
- Predict house prices in Boston using numeric and categorical features.
- Employ data scaling and linear regression.

**What I Did:**  
- Loaded data with fetch_openml.
- Cleaned and split the dataset.
- Used StandardScaler and LinearRegression with scikit-learn’s Pipeline.
- Evaluated model (R², MAE, MSE).
- Plotted histograms of targets, and scatterplots of actual vs. predicted prices.

**Results:**  
- Calculated and displayed train/test set R², MAE, and MSE.
- Histogram gave an overview of value ranges.
- Scatterplot showed model fit and residual trends.

---

### 3. **Random Forest & Decision Tree Regression on Diabetes (with GridSearchCV)**

**Objective:**  
- Compare Random Forest and Decision Tree for predicting diabetes progression.
- Use GridSearchCV to tune hyperparameters for best DecisionTreeRegressor.

**What I Did:**  
- Trained RandomForestRegressor and DecisionTreeRegressor.
- Tuned the decision tree with a grid search over hyperparameters.
- Compared MAE, MSE, and R² for both models.
- Visualized prediction distributions, actual vs. predicted, and feature importances.

**Results:**  
- Identified the best Decision Tree parameters.
- Random Forest generally performed better (higher R², lower errors).
- Feature importance plotted and top predictors revealed.
- Rich visual analysis of model predictions and errors.

---

### 4. **Residual Analysis (Diabetes Regression)**

**Objective:**  
- Assess regression diagnostics and model assumptions.

**What I Did:**  
- Plotted residuals vs. predictions.
- Generated boxplots and QQ-plots of residuals.

**Results:**  
- Visualized residual distribution shape and normality.
- Checked for homoscedasticity and outliers.

---

### 5. **Random Forest Classification with Hyperparameter Tuning on Breast Cancer**

**Objective:**  
- Classify tumors (malignant/benign) using Breast Cancer Wisconsin dataset.
- Use RandomForestClassifier and optimize with GridSearchCV.

**What I Did:**  
- Loaded the dataset and split into train/test sets.
- Trained a Random Forest model.
- Performed GridSearchCV for hyperparameter tuning.
- Evaluated with accuracy, classification report, and confusion matrix.
- Plotted the confusion matrix as a heatmap.

**Results:**  
- Achieved high classification accuracy.
- Best parameters identified from grid search.
- Clear confusion matrix visualized model performance by class.

---

## Summary

This project gave hands-on experience in applying, evaluating, and tuning machine learning models for both regression and classification on real data.  
Key skills demonstrated:
- Building and evaluating predictive models.
- Performing hyperparameter optimization.
- Using visualizations for insights.
- Practicing model diagnostics.

These templates can be expanded for any supervised ML problem—just plug in your dataset and tweak the pipeline!

---

## Author

Okanlawon Micheal Olatunji