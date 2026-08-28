# 🏠 California Housing Price Prediction using Random Forest Regression

A machine learning regression project that predicts **median house values** using the California Housing dataset and a **Random Forest Regressor**.

The project demonstrates a complete machine learning workflow — from data loading and preprocessing to model training, evaluation, feature importance analysis, prediction visualization, and residual analysis.

---

## 📌 Project Overview

Housing prices depend on several factors such as location, median income, number of rooms, population, household size, and proximity to the ocean.

The objective of this project is to build a regression model that learns these relationships and predicts the **median house value** for a given area.

### 🎯 Objective

> Predict `median_house_value` using housing, demographic, geographic, and income-related features.

---

## 📊 Dataset

The project uses the **California Housing Prices** dataset.

### Dataset Size

* **20,640 observations**
* **13 input features after preprocessing**
* Target variable: `median_house_value`

### Original Features

| Feature              | Description                             |
| -------------------- | --------------------------------------- |
| `longitude`          | Longitude of the housing district       |
| `latitude`           | Latitude of the housing district        |
| `housing_median_age` | Median age of houses in the district    |
| `total_rooms`        | Total number of rooms                   |
| `total_bedrooms`     | Total number of bedrooms                |
| `population`         | Population of the district              |
| `households`         | Number of households                    |
| `median_income`      | Median income of households             |
| `ocean_proximity`    | Location category relative to the ocean |
| `median_house_value` | Target house value                      |

---

## 🧹 Data Preprocessing

### 1. Missing Values

The `total_bedrooms` feature contained missing values.

Instead of deleting those observations, missing values were replaced using the **median** of the feature.

```python
df["total_bedrooms"] = df["total_bedrooms"].fillna(
    df["total_bedrooms"].median()
)
```

Median imputation was selected because it is less sensitive to extreme values than the mean.

### 2. Categorical Encoding

`ocean_proximity` is a categorical feature, so it cannot be directly used by the regression model.

It was converted using **One-Hot Encoding**:

```python
x = pd.get_dummies(
    x,
    columns=["ocean_proximity"],
    drop_first=True,
    dtype=int
)
```

This transformed the categorical variable into numerical indicator features.

After preprocessing, the feature matrix contained **13 input features**.

### 3. Train-Test Split

The dataset was divided into training and testing sets.

```python
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=42
)
```

The model learns from the training data and is evaluated on unseen test data.

---

# 🌲 Random Forest Regression

The final model used in this project is a **Random Forest Regressor**.

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees.

Instead of relying on a single decision tree, the model creates multiple trees and combines their predictions to produce a more robust result.

### Model Configuration

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=50,
    random_state=42
)

model.fit(x_train, y_train)
```

### Parameters

| Parameter      | Value | Purpose                    |
| -------------- | ----: | -------------------------- |
| `n_estimators` |    50 | Number of decision trees   |
| `random_state` |    42 | Makes results reproducible |

---

# 📏 Model Evaluation

The Random Forest model is evaluated using three regression metrics.

## R² Score

R² measures how much of the variation in house prices is explained by the model.

A value closer to **1** indicates stronger explanatory performance.

```python
r2_score(y_test, y_pred_rf)
```

---

## Mean Absolute Error — MAE

MAE measures the average absolute difference between actual and predicted house values.

```python
mean_absolute_error(y_test, y_pred_rf)
```

A lower MAE indicates smaller average prediction errors.

---

## Root Mean Squared Error — RMSE

RMSE measures prediction error while giving greater weight to larger errors.

```python
mean_squared_error(y_test, y_pred_rf) ** 0.5
```

A lower RMSE indicates better predictive performance.

---

# 🔎 Feature Importance

Random Forest provides a feature importance score for every input feature.

```python
importance = pd.DataFrame({
    "Feature": x.columns,
    "Importance": model.feature_importances_
}).sort_values(
    "Importance",
    ascending=False
)

display(importance)
```

Feature importance helps answer:

> **Which features contribute most to the Random Forest's predictions?**

This is different from a linear regression coefficient.

A higher Random Forest importance means that the feature contributed more to the model's tree-based decision process.

---

# 📈 Visualizations

## 1. Feature Importance

A feature-importance plot is used to identify the most influential variables in the model.

```python
plt.barh(x.columns, model.feature_importances_)
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.show()
```

---

## 2. Actual vs Predicted

The model's predictions are compared with the actual house values.

```python
plt.scatter(y_test, y_pred_rf, alpha=0.4)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linewidth=3
)

plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Actual vs Predicted - Random Forest Regressor")
plt.show()
```

The red diagonal line represents **perfect predictions**.

* Points close to the line → better predictions
* Points far from the line → larger prediction errors

---

# 📉 Residual Analysis

A residual represents the difference between the actual and predicted value:

```text
Residual = Actual Value - Predicted Value
```

Residuals were analyzed to understand the model's prediction errors.

```python
residuals = y_test - y_pred_rf
```

### Residual Plot

```python
plt.scatter(y_pred_rf, residuals, alpha=0.4)

plt.axhline(
    0,
    color="red",
    linestyle="--",
    linewidth=3
)

plt.xlabel("Predicted House Value")
plt.ylabel("Residual")
plt.title("Residual Plot - Random Forest Regressor")
plt.show()
```

The red horizontal line represents **zero residual**.

Ideally, residuals should be distributed around zero without a strong systematic pattern.

---

# 🛠️ Technologies Used

* **Python**
* **Pandas** — data manipulation
* **NumPy** — numerical operations
* **Matplotlib** — visualization
* **Seaborn** — statistical visualization
* **Scikit-learn** — machine learning
* **Google Colab** — development environment
* **Kaggle** — dataset source

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/california-housing-random-forest.git
cd california-housing-random-forest
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

# 🚀 Project Workflow

```text
California Housing Dataset
          ↓
Data Loading
          ↓
Data Inspection
          ↓
Missing Value Handling
          ↓
Median Imputation
          ↓
One-Hot Encoding
          ↓
Train-Test Split
          ↓
Random Forest Regressor
          ↓
Prediction
          ↓
Model Evaluation
          ↓
Feature Importance
          ↓
Actual vs Predicted
          ↓
Residual Analysis
```

---

# 📁 Project Structure

```text
california-housing-random-forest/
│
├── california_housing.ipynb
├── housing.csv
├── README.md
│
└── models/
    └── random_forest_model.pkl
```

> The exact files in the repository may vary depending on the final project version.

---

# 💾 Saving the Model

The trained model can be saved using Joblib:

```python
import joblib

joblib.dump(model, "random_forest_model.pkl")
```

The saved model can later be loaded without retraining:

```python
model = joblib.load("random_forest_model.pkl")
```

---

# 🔮 Making a Prediction

Once the model has been trained, predictions can be generated using:

```python
y_pred_rf = model.predict(x_test)
```

For a new observation, the input must first undergo the same preprocessing used during training before being passed to the model.

---

# 🧠 Key Concepts Demonstrated

This project demonstrates practical understanding of:

* Regression
* Supervised Machine Learning
* Missing-value handling
* Median imputation
* Categorical encoding
* Train-test splitting
* Ensemble learning
* Random Forest Regression
* Model evaluation
* R² Score
* MAE
* RMSE
* Feature importance
* Prediction analysis
* Residual analysis
* Model persistence

---

# 🎯 Why Random Forest?

Random Forest was selected as the final model because housing prices can depend on **complex and non-linear relationships** between features.

A Random Forest can capture interactions and non-linear patterns that a simple linear model may not represent effectively.

The ensemble of multiple decision trees also makes the model more robust than relying on a single tree.

---

# ⚠️ Limitations

The model has several limitations:

* It is trained only on the available California Housing dataset.
* Predictions depend on the quality and representativeness of the training data.
* Feature importance does not necessarily imply causation.
* Extremely high or unusual house values may still produce larger errors.
* The model does not include every factor that can influence real-world house prices.

---

# 🔮 Future Improvements

Possible future improvements include:

* Hyperparameter tuning
* Cross-validation
* Testing different numbers of trees
* Optimizing `max_depth`, `min_samples_split`, and other parameters
* Comparing additional ensemble algorithms
* Building a prediction interface
* Deploying the model as an API
* Creating an interactive web application
* Adding geographic visualization of housing prices

---

# 👨‍💻 Author

**Sudhanshu Ranjan**

Machine Learning Project — California Housing Price Prediction

---

## ⭐ Project Summary

This project demonstrates a complete **Random Forest Regression workflow** for California housing price prediction, including data preprocessing, model training, evaluation, feature importance analysis, prediction visualization, and residual diagnostics.

The primary goal is not only to predict house values but also to understand **how a Random Forest Regressor learns from housing data and which features contribute most to its predictions.**
