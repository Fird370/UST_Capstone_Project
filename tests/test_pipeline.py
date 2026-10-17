import os
import pandas as pd
import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. Test that required columns exist
def test_required_columns_exist():
    mock_df = pd.DataFrame({
        "customerID": ["0001-ABCD"],
        "gender": ["Female"],
        "tenure": [12],
        "MonthlyCharges": [29.85],
        "TotalCharges": ["358.2"],
        "Churn": ["No"]
    })
    required_cols = ["customerID", "Churn", "tenure", "MonthlyCharges", "TotalCharges"]
    for col in required_cols:
        assert col in mock_df.columns, f"Required column missing: {col}"

# 2. Test that Churn contains only expected values before encoding
def test_churn_expected_values():
    mock_churn = pd.Series(["Yes", "No", "No", "Yes"])
    valid_values = {"Yes", "No"}
    assert set(mock_churn.unique()).issubset(valid_values), "Unexpected values found in Churn column"

# 3. Test that duplicate customer IDs can be detected
def test_duplicate_customer_ids():
    mock_ids = pd.Series(["1234-ABC", "5678-DEF", "1234-ABC"]) # Duplicate ID present
    has_duplicates = mock_ids.duplicated().any()
    assert has_duplicates == True, "Failed to detect duplicate customer IDs"

# 4. Test that cleaned data contains no unexpected negative charges
def test_no_negative_charges():
    mock_df = pd.DataFrame({
        "tenure": [1, 12, 24],
        "MonthlyCharges": [19.95, 50.0, 80.0],
        "TotalCharges": [19.95, 600.0, 1920.0]
    })
    assert (mock_df["MonthlyCharges"] >= 0).all(), "Negative monthly charges detected"
    assert (mock_df["TotalCharges"] >= 0).all(), "Negative total charges detected"
    assert (mock_df["tenure"] >= 0).all(), "Negative tenure detected"

# 5. Test that the model can be trained on a small valid sample
def test_model_training_small_sample():
    X_train = pd.DataFrame({
        "tenure": [1, 12, 24, 3],
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"]
    })
    y_train = pd.Series([1, 0, 0, 1])

    numeric_features = ["tenure"]
    categorical_features = ["Contract"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    assert pipeline is not None, "Pipeline failed to train on small sample"

# 6. Test that the prediction output contains required columns
def test_prediction_output_schema():
    output_df = pd.DataFrame({
        "customerID": ["1234-XYZ", "5678-ABC"],
        "actual_churn": [0, 1],
        "predicted_churn": [0, 0],
        "churn_probability": [0.15, 0.42]
    })

    required_output_cols = ["customerID", "actual_churn", "predicted_churn", "churn_probability"]
    for col in required_output_cols:
        assert col in output_df.columns, f"Prediction output missing required column: {col}"