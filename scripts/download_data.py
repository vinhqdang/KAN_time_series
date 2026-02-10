
import os
import sys
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import load_multivariate_data
from src.config import START_DATE, END_DATE

def main():
    print(f"Downloading Financial Data from {START_DATE} to {END_DATE}...")
    try:
        _, dataset = load_multivariate_data()
        df = dataset['raw_df']
        
        output_dir = 'data'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'financial_2020_2025.csv')
        
        df.to_csv(output_path)
        print(f"Data saved to {output_path}")
        print("Columns:", df.columns.tolist())
        print("Shape:", df.shape)
        
    except Exception as e:
        print(f"Error downloading data: {e}")

if __name__ == "__main__":
    main()
