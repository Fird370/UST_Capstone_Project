import pandas as pd
import os

def run_data_validation(file_path):
    #load dataset
    df = pd.read_csv(file_path)
    
    print("----Data set type----")
    rows, cols = df.shape
    print(f"Dataset Shape (Rows, Columns): {df.shape}")

    print("\n")
    print("----First 5 Rows----")
    print(df.head())

    print("\n")
    print("----Column Data type----")
    print(df.info())

    print("\n")
    # check for Duplicate customer IDs
    duplicates = df['customerID'].duplicated().sum() if 'customerID' in df.columns else 0
    print(f"Duplicate customer IDs: {duplicates}")

    # Identify Missing-value fields
    # Identifies columns that contain at least one missing/null value
    missing_cols = df.columns[df.isna().any()].tolist()

    # Check for hidden blank spaces in object columns (like TotalCharges)
    blank_space_cols = []
    for col in df.select_dtypes(include=['object', 'str']).columns:
        if (df[col].astype(str).str.strip() == '').any():
            blank_space_cols.append(col)

      
    # Combine actual NaNs and blank spaces for a complete missing-fields list
    all_missing_fields = list(set(missing_cols + blank_space_cols))
    missing_str = ", ".join(all_missing_fields) if all_missing_fields else "None"
    print(f"Missing-value fields: {missing_str}")

    # TotalCharges conversion issues
    if 'TotalCharges' in df.columns:
        converted_total = pd.to_numeric(df['TotalCharges'], errors='coerce')
        total_charges_issues = converted_total.isna().sum()
    else:
        total_charges_issues = "Column missing"

    print(f"TotalCharges conversion issues: {total_charges_issues} (blank spaces handled)")

    # Negative numeric values check for MonthlyCharges & TotalCharges 
    negative_count = 0
    if 'MonthlyCharges' in df.columns:
        negative_count += (df['MonthlyCharges'] < 0).sum()
    if 'TotalCharges' in df.columns:
        numeric_total = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        negative_count += (numeric_total < 0).sum()
    print(f"Negative numeric values: {negative_count}")

    # Check Churn target labels
    if 'Churn' in df.columns:
        unexpected_churn = df[~df['Churn'].isin(['Yes', 'No'])].shape[0]
        invalid_target = "PASS" if unexpected_churn == 0 else f"REVIEW ({unexpected_churn} invalid rows)"
    else:
        invalid_target = "N/A"
    print(f"Invalid target values: {invalid_target}")

    # Tenure range check (expected between 0 and 72 months)
    if 'tenure' in df.columns:
        invalid_tenure = ((df['tenure'] < 0) | (df['tenure'] > 72)).sum()
        tenure_status = "PASS" if invalid_tenure == 0 else f"REVIEW ({invalid_tenure} out of range)"
    else:
        tenure_status = "N/A"

    print(f"Tenure range check: {tenure_status}")

    # Overall Status Decision Logic
    # We expect exactly 11 conversion issues in TotalCharges (the famous blank spaces for tenure=0)
    overall_status = "PASS" if total_charges_issues == 11 and negative_count == 0 and duplicates == 0 else "REVIEW"

    print(f"Overall status: {overall_status}")

    return df

if __name__ == "__main__":
    raw_path = os.path.join("data", "raw", "Telco-Customer-Churn.csv")
    
    try:
        run_data_validation(raw_path)
    except FileNotFoundError:
        print(f"[ERROR] Could not find raw file at '{raw_path}'. Please ensure the CSV is placed correctly.")