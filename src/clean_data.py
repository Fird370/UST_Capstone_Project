import pandas as pd

def load_and_inspect_data(file_path):
    df = pd.read_csv(file_path)

    print("----Data set type----")
    print(f"Dataset Shape (Rows, Columns): {df.shape}")

    print("----First 5 Rows----")
    print(df.head())

    print("----Column Data type----")
    print(df.info())

    return df

if __name__ == "__main__":
    path = "data/raw/Telco-Customer-Churn.csv"
    load_raw_data(path)