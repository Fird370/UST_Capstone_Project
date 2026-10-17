This is my data science capstone project where I process and clean telecom customer data to predict customer churn (whether a customer stays or leaves). 

The main goal is to build clean, modular Python scripts to check data quality, clean missing values, run automated tests, and prepare everything for machine learning.

---

## Project Structure

```text
telco-churn-capstone/
├── data/
│   ├── raw/                  <-- Original CSV file (never edited or modified)
│   └── processed/            <-- Output files created by my scripts (cleaned data & IDs)
├── src/                      <-- Python source code scripts
├── tests/                    <-- Automated unit and integration tests
├── output/                   <-- Final model results and prediction exports
├── .github/                  
│   └── workflows/            <-- CI/CD pipeline automation settings
├── main.py                   <-- Main script to run the pipeline
├── requirements.txt          <-- Required Python libraries list
├── README.md                 <-- Project documentation
└── .gitignore                <-- Files and folders ignored by Git
How to Set Up and Run
Install dependencies:

Bash
pip install -r requirements.txt
Put the raw dataset: Place Telco-Customer-Churn.csv inside the data/raw/ folder.

Run the validation script:

Bash
python src/validate_data.py
Run the cleaning script:

Bash
python src/clean_data.py
What My Scripts Do
1. Data Validation (src/validate_data.py)
This script loads the raw CSV file and checks for data quality issues before cleaning. It prints out:

Dataset shape and first 5 rows.

Column names and data types (info()).

Check for duplicate customer IDs.

Missing values and blank spaces in TotalCharges.

Checks for negative numbers in charges.

Checks if Churn and tenure values are valid.

2. Data Cleaning Decisions (src/clean_data.py)
Here are the exact changes made to prepare the data for machine learning and why:

Separated customerID:

What changed: Extracted customerID and saved it separately to data/processed/customer_ids.csv, then dropped the column from the main dataset.

Why: Customer IDs are just random text strings and do not help the model learn general churn patterns. Keeping them separate allows us to match model predictions back to specific customers later.

Fixed Missing Values in TotalCharges:

What changed: Converted TotalCharges to numbers, found 11 missing blank spaces, and filled them with 0.0.

Why: The missing values belong to brand new customers with tenure = 0. Since they haven't been billed yet, 0.0 is logical and avoids deleting those rows.

Converted Binary Columns (Yes/No to 1/0):

What changed: Mapped Partner, Dependents, PhoneService, PaperlessBilling, and Churn from "Yes"/"No" to 1 and 0.

Why: Machine learning models require numerical inputs instead of text labels.

One-Hot Encoding for Categorical Columns:

What changed: Used pd.get_dummies(df, drop_first=True) to convert multi-option text columns (like Contract and PaymentMethod) into numeric indicator columns.

Why: Converts categories into numbers so algorithms can read them, while drop_first=True avoids duplicate mathematical information.
