import pandas as pd
import os

def run_eda(raw_file_path, output_dir):
    df = pd.read_csv(raw_file_path)

    # Overall Churn Rate
    churn_rate = (df['Churn'] == 'Yes').mean() * 100
    print(f"\n1. Overall Churn Rate: {churn_rate:.2f}%")
    obs_1 = f"Approximately {churn_rate:.2f}% of customers left the service."
    print(f"   -> Observation: {obs_1}")

    # Churn by Contract Type
    print("\n2. Churn Rate by Contract Type:")
    contract = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    for k, v in contract.items():
        print(f"   - {k}: {v:.2f}%")
    obs_2 = f"{contract.idxmax()} contracts show the highest churn at {contract.max():.2f}%."
    print(f"   -> Observation: {obs_2}")

    # Churn by Internet Service
    print("\n3. Churn Rate by Internet Service:")
    internet = df.groupby('InternetService')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    for k, v in internet.items():
        print(f"   - {k}: {v:.2f}%")
    obs_3 = f"Customers with {internet.idxmax()} internet exhibit the highest churn at {internet.max():.2f}%."
    print(f"   -> Observation: {obs_3}")

    # Churn by Payment Method
    print("\n4. Churn Rate by Payment Method:")
    payment = df.groupby('PaymentMethod')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    for k, v in payment.items():
        print(f"   - {k}: {v:.2f}%")
    obs_4 = f"Customers using {payment.idxmax()} show an increased churn rate of {payment.max():.2f}%."
    print(f"   -> Observation: {obs_4}")

    # Churn by Tech Support
    print("\n5. Churn Rate by Tech Support Availability:")
    tech = df.groupby('TechSupport')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    for k, v in tech.items():
        print(f"   - {k}: {v:.2f}%")
    obs_5 = f"Customers without tech support show a higher churn rate of {tech.get('No', 0):.2f}%."
    print(f"   -> Observation: {obs_5}")

    #Save the results to the text file
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "eda_summary.txt")

    with open(report_path, "w") as f:
        f.write("TASK 5: EXPLORATORY DATA ANALYSIS (EDA) SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"1. Overall Churn Rate: {churn_rate:.2f}%\n")
        f.write(f"   -> Observation: {obs_1}\n\n")
        
        f.write("2. Churn Rate by Contract Type:\n")
        for k, v in contract.items():
            f.write(f"   - {k}: {v:.2f}%\n")
        f.write(f"   -> Observation: {obs_2}\n\n")
        
        f.write("3. Churn Rate by Internet Service:\n")
        for k, v in internet.items():
            f.write(f"   - {k}: {v:.2f}%\n")
        f.write(f"   -> Observation: {obs_3}\n\n")
        
        f.write("4. Churn Rate by Payment Method:\n")
        for k, v in payment.items():
            f.write(f"   - {k}: {v:.2f}%\n")
        f.write(f"   -> Observation: {obs_4}\n\n")
        
        f.write("5. Churn Rate by Tech Support Availability:\n")
        for k, v in tech.items():
            f.write(f"   - {k}: {v:.2f}%\n")
        f.write(f"   -> Observation: {obs_5}\n")

    print("\n" + "="*50)
    print(f"EDA summary report saved safely to: {report_path}")

if __name__ == "__main__":
    raw_path = os.path.join("data", "raw", "Telco-Customer-Churn.csv")
    output_directory = "outputs"
    
    try:
        run_eda(raw_path, output_directory)
    except FileNotFoundError:
        print(f"Error:Could not find raw file at '{raw_path}'. Please check your file path.")