import pandas as pd
import os

def clean_telco_data(raw_file_path, processed_dir):
    df = pd.read_csv(raw_file_path)

    #drop customerID to make machine learning easier
    if 'customerID' in df.columns:
        customer_ids = df[['customerID']].copy()
        id_output_path = os.path.join(processed_dir, "customer_ids.csv")
        customer_ids.to_csv(id_output_path, index=False)
        print(f"Retained customer IDs saved safely to: {id_output_path}")

        df = df.drop(columns=['customerID'])
        print("Dropped 'customerID' column.")


    #Change NaN value to 0 in TotalCharges column
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        missing_count = df['TotalCharges'].isna().sum()
        print(f"Found {missing_count} missing values in 'TotalCharges'")
        df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
        print(f"Handled {missing_count} missing values in 'TotalCharges' by filling with 0.0.")


    #change Yes/No to binary 1/0
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    print("change Yes/No to 1 and 0.")

    #One-Hot Encoding for remaining categorical text columns
    #drop_first=True prevents multi-collinearity (redundancy in math)
    df = pd.get_dummies(df, drop_first=True)
    print("Applied One-Hot Encoding to categorical columns.")

    print(f"Final Cleaned Shape: {df.shape}")

    return df

if __name__ == "__main__":
    raw_path = os.path.join("data", "raw", "Telco-Customer-Churn.csv")
    processed_dir = "data/processed"
    
    # Ensure processed directory exists
    os.makedirs(processed_dir, exist_ok=True)
    
    output_path = os.path.join(processed_dir, "cleaned_churn.csv")
    
    try:
        cleaned_df = clean_telco_data(raw_path, processed_dir)
        cleaned_df.to_csv(output_path, index=False)
        print(f"Cleaned feature matrix saved safely to: {output_path}")
    except FileNotFoundError:
        print(f"Could not find raw file at '{raw_path}'. Please check your file path.")