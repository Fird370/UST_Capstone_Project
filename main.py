import os
import sys

# Ensure Python can find modules inside the 'src' directory
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.train_model import train_and_evaluate

def main():
    print("STARTING UST CAPSTONE: TELCO CHURN PREDICTION PIPELINE")
    
    # Define directories based on your project structure
    data_directory = os.path.join("data", "processed")
    output_directory = "outputs"  # Matches your root 'outputs/' directory
    
    try:
        print("\nRunning model training and evaluation...")
        train_and_evaluate(data_directory, output_directory)
        
        print("\n")
        print("Master pipeline completed successfully!")
        print(f" Check your results inside the '{output_directory}/' folder.")
        
    except Exception as e:
        print(f"\n Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()