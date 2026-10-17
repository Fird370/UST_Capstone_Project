# Telco Churn Prediction Pipeline (UST Capstone Project)

## Project Title
End-to-End Telco Customer Churn Prediction and Software Packaging System

---

## Business Problem
Customer churn—when subscribers leave for a competitor—results in significant revenue loss. Acquiring new customers is more costly than retaining current ones. This project builds a predictive machine learning model to identify high-risk customers early, allowing the retention team to intervene with targeted offers.

---

## Dataset Source and Attribution
* **Dataset Name:** Telco Customer Churn
* **Source:** IBM Sample Data Sets / Publicly available via Kaggle.
* **Attribution:** Provided as part of the UST Capstone Project curriculum.

---

## Dataset Overview
* **Total Records:** 7,043 customer instances.
* **Features:** Customer demographics, account service subscriptions (such as streaming and internet type), payment methods, tenure, monthly charges, and total charges.
* **Target Variable:** Churn (Binary: Yes/No indicating whether the customer left within the last month).

---

## Project Architecture
text
UST_Capstone_project/
├── data/
│   ├── raw/                  <-- Original immutable raw dataset
│   └── processed/            <-- Cleaned data files and customer IDs
│       ├── cleaned_churn.csv
│       └── customer_ids.csv
├── src/                      <-- Python source code modules
│   ├── __init__.py
│   ├── clean_data.py         <-- Raw data cleaning
│   ├── validate_data.py      <-- Schema and structural checks
│   ├── eda_analysis.py       <-- Exploratory data analysis scripts
│   ├── preprocessing.py      <-- Feature and target preparation
│   ├── model.py              <-- ColumnTransformer and Pipeline definition
│   ├── evaluation.py         <-- Metric calculation and output generation
│   ├── train_model.py        <-- Training coordinator
│   └── pipeline.py           <-- End-to-end pipeline orchestration
├── tests/                    <-- Automated unit test suite
│   └── test_pipeline.py
├── output/                   <-- Final model results and prediction exports
│   └── predictions.csv
├── .github/
│   workflows/
│       └── ci.yml            <-- GitHub Actions CI/CD workflow
├── main.py                   <-- Master entry point
├── requirements.txt          <-- Python dependencies
└── README.md                 <-- Project documentation

## Installation Instructions
Clone the repository:

Bash
git clone [https://github.com/your-username/UST_Capstone_project.git](https://github.com/your-username/UST_Capstone_project.git)
cd UST_Capstone_project
Install the required dependencies:

Bash
pip install -r requirements.txt

## How to Run the Application
Run the entire machine learning pipeline using the master script:

Bash
python main.py

This executes data loading, validation, preprocessing, model training, evaluation, and saves the final predictions to output/predictions.csv.

## How to Run Tests
Run the automated unit test suite using pytest:

Bash
pytest -v

## Validation Checks Performed
Required Columns Check: Ensures all critical features and targets exist.

Target Value Validation: Verifies that target columns contain only expected classes.

Duplicate Detection: Scans customer IDs to detect duplicate records.

Negative Value Safeguards: Confirms financial and tenure metrics contain no invalid negative values.

## Cleaning Decisions
TotalCharges Conversion: Missing or blank space entries for brand-new customers were converted to numeric floats and filled with zero.

Identifier Isolation: Customer IDs were separated from the feature matrix to prevent data leakage while retaining them for output mapping.

Target Encoding: The categorical target column was mapped to binary integers (1/0).

## Features Used by the Model
Numeric Features: tenure, MonthlyCharges, TotalCharges (scaled using StandardScaler).

Categorical Features: Contract type, Payment method, Internet service, and add-on services (encoded using OneHotEncoder).

## Model Used
Logistic Regression, implemented via scikit-learn Pipeline and ColumnTransformer to prevent data leakage during scaling and encoding.

## Evaluation Metrics
Accuracy: 80.70%

Precision: 66.04%

Recall: 56.15%

F1 Score: 0.6069

## Exploratory Data Analysis (EDA) Findings
Month-to-month contract holders show significantly higher churn rates than multi-year contract holders.

Newer customers with low tenure are more prone to churning than long-term customers.

Fiber optic internet users exhibit higher churn rates compared to DSL users.

Higher monthly billing amounts correlate with increased churn likelihood, indicating price sensitivity.

## Limitations
Single Model Used: I only used Logistic Regression for this project, so I haven't tested how other types of machine learning models would perform on this dataset.

## Future Improvements
Try Different Models: Experiment with other machine learning algorithms (like Decision Trees or Random Forests) to see if they yield higher accuracy.

## Example Output (output/predictions.csv)
customerID,actual_churn,predicted_churn,churn_probability
7590-VHVEG,0,0,0.1245
5575-GNVDE,0,0,0.3812
3668-QPYBK,1,1,0.7420