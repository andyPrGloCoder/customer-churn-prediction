# Customer Churn Prediction

## Project Description
This project builds an end-to-end logistic regression model to predict customer churn for a telecommunications company using the Telco Customer Churn dataset. The workflow includes exploratory data analysis, missing-value cleanup, feature engineering, preprocessing with a scikit-learn pipeline, hyperparameter tuning with GridSearchCV, model evaluation, coefficient interpretation, and a deployment simulation using a saved model.

## How to Run the Code
1. Open `Andy_IN503_Unit10_Assignment.ipynb` in Google Colab.
2. Run the notebook cells from top to bottom.
3. The notebook installs `kagglehub` and downloads the public Telco Customer Churn dataset automatically.
4. GridSearchCV trains and tunes the logistic regression model.
5. The final pipeline is saved as `customer_churn_model.pkl`.
6. The final cells load the saved model and demonstrate churn predictions for three example customers.

## Key Findings Summary
- The original dataset contains 7,043 customers and 21 columns.
- Eleven blank values in `TotalCharges` are converted to missing values and removed, leaving 7,032 records.
- About 26.6% of customers churned, so the target variable is moderately imbalanced.
- Month-to-month customers have substantially higher churn than customers on one-year or two-year contracts.
- Fiber optic customers show higher churn than DSL or customers without internet service.
- Customers with shorter tenure are more likely to churn.
- The tuned logistic regression model produced approximately 80.5% test accuracy, 65.1% precision, 57.2% recall, a 60.9% F1-score, and an ROC AUC of about 0.84 in validation testing with `random_state=42`.
- Strong model effects include contract type, tenure, and fiber optic internet service.

## Dependencies
- Python 3
- pandas
- NumPy
- matplotlib
- seaborn
- scikit-learn
- joblib
- kagglehub

## Model Output
Running the notebook creates:
- `customer_churn_model.pkl` - saved preprocessing and logistic regression pipeline
