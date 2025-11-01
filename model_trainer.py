# model_trainer.py (FINAL CORRECTED VERSION)
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load # Ensure these are imported
import joblib

# Define the exact features used for training and prediction (MUST BE GLOBAL)
FEATURE_COLS = ['SR_bot', 'SR_human', 'AGE_Score', 'p_distort', 'p_noise', 'p_rotate']

def train_model(data_path="data/sample_data.csv"):
    """Loads full simulation data, calculates AGE_Score, and trains the model."""
    os.makedirs("models", exist_ok=True)
    
    # 1. Load the full raw simulation data 
    if not os.path.exists(data_path):
        print(f"[ERROR] Data file not found at {data_path}. Run bot.py first!")
        return None, None
        
    df_raw = pd.read_csv(data_path)
    print(f"[INFO] Loaded {len(df_raw)} raw entries from simulation.")
    
    # 2. AGGREGATION: Calculate AGE_Score per unique parameter set
    df_bot = df_raw[df_raw['typing_time'] == 0]
    df_human = df_raw[df_raw['typing_time'] > 0]

    group_cols = ['p_distort', 'p_noise', 'p_rotate', 'generation_strategy']
    
    sr_bot = df_bot.groupby(group_cols)['success'].mean().rename('SR_bot')
    sr_human = df_human.groupby(group_cols)['success'].mean().rename('SR_human')

    df_aggregated = pd.merge(
        sr_bot.reset_index(), 
        sr_human.reset_index(), 
        on=group_cols,
        how='inner'
    )

    df_aggregated['AGE_Score'] = df_aggregated['SR_human'] - df_aggregated['SR_bot']
    df_aggregated['Target'] = (df_aggregated['AGE_Score'] > 0).astype(int) 

    # --- 3. Training Data Setup ---
    # CRITICAL: Use the standardized feature list
    X = df_aggregated[FEATURE_COLS]
    y = df_aggregated["Target"]

    print(f"[INFO] Aggregated data reduced to {len(df_aggregated)} unique parameter configurations for training.")
    
    # 4. Train the Classifier (Ensure 'model' is defined here)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, os.path.join("models", "rf_defense_model.pkl")) # Use os.path.join for saving
    print("[+] Random Forest classifier trained and saved → models/rf_defense_model.pkl")
    
    # 5. Return the defined variables
    return model, df_aggregated