# test_pipeline_phase1.py
import os
import pandas as pd

# Make sure Python can find your src package
import sys
sys.path.append(os.path.abspath("src"))

from data.load_data import load_data
from data.preprocess import preprocess_data
from features.build_features import build_features

# === CONFIG ===
DATA_PATH = "D:\\ML\\Telco Customer Churn ML\\test\\data\\raw\\Telco-Customer-Churn.csv"  
TARGET_COL = "Churn"

def main():
    print("=== Testing Phase 1: Load → Preprocess → Build Features === - test_pipeline_phase1_data_features.py:18")

    # 1. Load Data
    print("\n[1] Loading data... - test_pipeline_phase1_data_features.py:21")
    df = load_data(DATA_PATH)
    print(f"Data loaded. Shape: {df.shape} - test_pipeline_phase1_data_features.py:23")
    print(df.head(3))

    # 2. Preprocess
    print("\n[2] Preprocessing data... - test_pipeline_phase1_data_features.py:27")
    df_clean = preprocess_data(df, target_col=TARGET_COL)
    print(f"Data after preprocessing. Shape: {df_clean.shape} - test_pipeline_phase1_data_features.py:29")
    print(df_clean.head(3))

    # 3. Build Features
    print("\n[3] Building features... - test_pipeline_phase1_data_features.py:33")
    df_features = build_features(df_clean, target_col=TARGET_COL)
    print(f"Data after feature engineering. Shape: {df_features.shape} - test_pipeline_phase1_data_features.py:35")
    print(df_features.head(3))

    print("\n✅ Phase 1 pipeline completed successfully! - test_pipeline_phase1_data_features.py:38")

if __name__ == "__main__":
    main()