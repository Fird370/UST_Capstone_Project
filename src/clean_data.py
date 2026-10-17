import pandas as pd
import os

def load_and_inspect_data(file_path):
    df = pd.read_csv(file_path)

    print("----Data set type----")
    print(f"Dataset Shape (Rows, Columns): {df.shape}")

    print("----First 5 Rows----")
    print(df.head())

    print("----Column Data type----")
    print(df.info())

    #Change NaN value to 0 in TotalCharges column
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        missing_count = df['TotalCharges'].isna().sum()
        print(f"Found {missing_count} missing values in 'TotalCharges'")
        df['TotalCharges'] = df['TotalCharges'].fillna(0.0)

    #drop customerID to make machine learning easier
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        print("Dropped 'customerID' column.")

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
    print("----Final Cleaned Data First 5 Rows----")
    print(df.head())

    return df

if __name__ == "__main__":
    raw_path = "data/raw/Telco-Customer-Churn.csv"
    processed_dir = "data/processed"
    output_path = os.path.join(processed_dir, "cleaned_churn.csv")
    
    os.makedirs(processed_dir, exist_ok=True)
    
    cleaned_df = load_and_inspect_data(raw_path)
    cleaned_df.to_csv(output_path, index=False)
    print(f"Cleaned data saved safely to: {output_path}")