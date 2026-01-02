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




# Machine Learning Analytics With scikit-learn Datasets on scikit_task.ipynb

A collection of mini-projects using classic datasets from scikit-learn for regression and classification.  
Each script demonstrates model training, hyperparameter tuning, metrics, and visualization.

---

## Objectives

- Build regression and classification models using scikit-learn.
- Evaluate model performance with common metrics.
- Visualize predictions, errors, class distributions, and feature importances.
- Use GridSearchCV for hyperparameter tuning.
- Gain hands-on experience with real-world data analysis pipelines.

---

## Datasets Used

- **Diabetes** (Regression, and Binned Classification)
- **Boston Housing** (Regression)
- **Wine** (Classification, 3 classes)
- **Digits** (Classification, 10 classes, images)
- **Breast Cancer Wisconsin** (Classification, 2 classes)

---

## Requirements

- Python 3.7+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

Install with:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

---

## Code Summaries, Objectives and Results

### 1. **Random Forest Classification on Binned Diabetes Scores**

**Objective:**  
- Bin diabetes score into 3 categories.
- Classify patients using RandomForestClassifier.
- Tune model via GridSearchCV.

**Key Steps:**  
- Data exploration and binning of targets.
- Model training and hyperparameter tuning.
- Evaluate with accuracy score, classification report, and confusion matrix.
- Visualize class distributions and compare actual vs predicted.

**Results:**  
- Model classified diabetes progression into bins with measurable accuracy.
- Clear confusion matrix and distribution plots for performance analysis.

---

### 2. **Random Forest Regression on Boston Housing**

**Objective:**  
- Predict house prices using regression.
- Assess model fit and prediction error.

**Key Steps:**  
- Train/test split and RandomForestRegressor training.
- Visualize actual vs predicted values and prediction distributions.
- Print model metrics: R², MAE, MSE.

**Results:**  
- Achieved reasonable R² scores.
- Visual scatterplot to check fit quality.
- Histogram of predicted prices for further analysis.

---

### 3. **Random Forest Classification on Wine Data**

**Objective:**  
- Classify wine into one of three types.
- Evaluate test accuracy and confusion matrix.

**Key Steps:**  
- Train RandomForestClassifier plus pipeline.
- Visualize predicted class distribution and confusion matrix.

**Results:**  
- Demonstrated high accuracy.
- Plots show how predictions distribute across wine classes.

---

### 4. **Random Forest and Logistic Regression on Digits Data**

**Objective:**  
- Classify handwritten digit images (0–9).
- Explore feature importances and residuals.
- Display grid-searched tuning results.
- Show sample predicted images.

**Key Steps:**  
- Train RandomForestClassifier and tune with GridSearchCV.
- Print best parameters and cross-validated accuracy.
- Visualize feature importances and prediction residuals.
- Display five random test digit images with predicted labels.

**Results:**  
- Showed strong classifier performance.
- Heatmaps highlight influential features.
- Sample plots of test images confirm class prediction visually.

---

### 5. **Random Forest Classification on Breast Cancer Data**

**Objective:**  
- Classify tumors as malignant or benign.
- Tune Random Forest classifier and visualize confusion matrix.

**Key Steps:**  
- Train and grid-search best estimator.
- Print best parameters, model metrics, confusion matrix.
- Display confusion matrix heatmap for class separation.

**Results:**  
- High test accuracy.
- Clear visual separation of true and predicted classes.

---

## Summary

These scripts provide templates for supervised machine learning using popular datasets, including data exploration, modeling, hyperparameter tuning, and robust result visualization.  
Use these as starting points for your own projects or data science practice.

---

## Author

Okanlawon Micheal Olatunji