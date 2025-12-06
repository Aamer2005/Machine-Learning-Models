# House Price Prediction Project

## Project Overview

This project predicts house prices based on property features using machine learning. It includes full data exploration, preprocessing, model training, evaluation, and hyperparameter tuning. The project demonstrates a complete machine learning workflow suitable for a resume or Kaggle publication.

## Dataset

* Source: Kaggle / Housedata CSV
* Columns include: `price`, `sqft_living`, `bedrooms`, `bathrooms`, `sqft_lot`, and more.
* Irrelevant features removed: `street`, `city`, `country`, `statezip`, `date`.

## Project Steps

1. **Problem Definition**

   * Predict house prices (regression problem).

2. **Data Loading**

   * Load dataset using Pandas.

3. **Data Understanding**

   * Check data types, missing values, summary statistics.

4. **Exploratory Data Analysis (EDA)**

   * Visualize distributions.
   * Use Matplotlib for all plots.

5. **Data Cleaning**

   * Remove irrelevant columns and handle missing values.

6. **Feature Engineering**

   * Not Required.

7. **Feature Selection**

   * Keep relevant numeric features for model.

8. **Train-Test Split**

   * Split data 80-20 for training and testing.

9. **Feature Scaling**

   * StandardScaler applied to features.

10. **Model Selection**

    * Models: Linear Regression, Random Forest, Decision Tree, KNN.

11. **Model Training**

    * Fit each model on training data.

12. **Model Evaluation**

    * Metrics: MSE, RMSE, MAE, R² Score.

13. **Model Comparison**

    * Compare all models to select the best.

14. **Hyperparameter Tuning**

    * Not Applied.

15. **Final Model Selection**

    * Best performing model selected based on RMSE & R².

16. **Documentation**

    * All steps explained and visualized using plots.

## Key Visualizations

* Histogram of prices

## Results

* Best model: LinearRegression
* Example Metrics:

name          Linear Regression
MSE         815231992444.915039
RMSE              902901.983853
ASE               193669.804998
r2_score               0.039208


## Future Improvements

* Add more advanced models like XGBoost or LightGBM
* Perform feature engineering and selection
* Use SHAP values for model interpretability
* Deploy model as a web app using Streamlit or Flask

## Requirements

* Python 3
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

## How to Run

1. Load `data.csv` into the project directory.
2. Run the Jupyter Notebook or Python script sequentially.
3. Check all visualizations and evaluation metrics.
4. Apply GridSearchCV to tune the Random Forest model.

---

**Author:** Mohammed Aamer
**Project Type:** Machine Learning Regression / House Price Prediction
