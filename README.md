This is my data science capstone project where I process and clean telecom customer data to predict customer churn (whether a customer stays or leaves). 

The main goal is to build clean, modular Python scripts to check data quality, clean missing values, run automated tests, and prepare everything for machine learning.

---

## Project Structure

```text
UST_Capstone_project/
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
This script loads the raw CSV file and checks for data quality issues before cleaning. It inspects:

Dataset shape and first 5 rows.

Column names and data types (info()).

Duplicate customer IDs, missing values, and blank spaces in TotalCharges.

Validity of target labels (Churn) and tenure ranges.

2. Data Cleaning (src/clean_data.py)
Here are the exact changes made to prepare the data for machine learning and why:

Separated customerID: Saved safely to data/processed/customer_ids.csv and dropped from the main dataset to prevent machine learning models from overfitting on random ID strings.

Fixed TotalCharges: Converted to numeric, identified blank spaces for new customers (tenure = 0), and filled missing values with 0.0.

Binary Encoding: Converted text columns like Partner, Dependents, and Churn from Yes/No to binary 1/0.

One-Hot Encoding: Applied pd.get_dummies with drop_first=True to multi-option text categories so machine learning algorithms can process them mathematically.

3. Exploratory Data Analysis (src/eda_analysis.py)
This script explores key churn drivers across the dataset using Pandas calculations and automatically generates dynamic observations:

Calculates overall churn percentage.

Breaks down churn rates across Contract types, Internet Services, Payment Methods, and Tech Support availability.

Automatically prints results to the terminal and saves a clean summary report inside outputs/eda_summary.txt.