import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_and_evaluate(data_dir, output_dir):
    cleaned_path = os.path.join(data_dir, "cleaned_churn.csv")
    ids_path = os.path.join(data_dir, "customer_ids.csv")
    
    df = pd.read_csv(cleaned_path)
    df_ids = pd.read_csv(ids_path)

    #defien feature
    if "customerID" in df.columns:
        X = df.drop(columns=["Churn", "customerID"])
    else:
        X = df.drop(columns=["Churn"])
        
    y = df["Churn"]
    customer_ids = df_ids["customerID"]

    #train/test split the model")
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, customer_ids,
        test_size=0.20,
        random_state=42,
        stratify=y #Ensure that both training set (y_train) and testing set (y_test) maintain that exact same ratio.
    )

    #Training Logistic Regression model")
    # Automatically detect column types from X_train
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Define the sorting machine (ColumnTransformer)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Define the assembly line (Pipeline) binding preprocessing + model together
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])

    # Train the entire pipeline (fitting scaler and encoder ONLY on X_train to prevent leakage!)
    model_pipeline.fit(X_train, y_train)

    #Generating predictions and evaluating performance
    predictions = model_pipeline.predict(X_test)
    probabilities = model_pipeline.predict_proba(X_test)[:, 1] # Probability of churning

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    print("\n")
    print("MODEL EVALUATION METRICS")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(matrix)

    print("\n")
    print("Confusion Matrix Legends:")
    print(f"Top Left: True Negatives  (Correctly predicted stay)")
    print(f"Top Right: False Positives (False alarms)")
    print(f"Bottom Left: False Negatives (Missed churners)")
    print(f"Bottom right: True Positives  (Successfully caught churners)")

    # Create outputs/predictions.csv
    # Save test results matching customer IDs, actuals, and risk scores.
    # -------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    predictions_path = os.path.join(output_dir, "predictions.csv")

    predictions_df = pd.DataFrame({
        "customerID": id_test,
        "actual_churn": y_test.values,
        "predicted_churn": predictions,
        "churn_probability": probabilities
    })

    predictions_df.to_csv(predictions_path, index=False)
    print("\n")
    print(f"Predictions saved safely to: {predictions_path}")

if __name__ == "__main__":
    data_directory = os.path.join("data", "processed")
    output_directory = "outputs"
    
    try:
        train_and_evaluate(data_directory, output_directory)
    except FileNotFoundError as e:
        print(f"{e}. Please ensure your processed data files exist.")