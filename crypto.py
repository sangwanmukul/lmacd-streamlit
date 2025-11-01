# crypto.py
import random
import string
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load

# --- Configuration ---
# FIX 1: Define the model file path inside the 'models' directory.
ML_MODEL_FILE = os.path.join("models", 'rf_defense_model.pkl') 
DEFAULT_ANALYSIS_FILE = os.path.join("data", "sample_data.csv")

def random_text(length=5):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def get_generation_params(strategy):
    """Returns challenge parameters for each strategy."""
    if strategy == "G1_Baseline":
        return {'p_distort': random.randint(2, 4),
                'p_noise': round(random.uniform(0.3, 0.5), 2),
                'p_rotate': 10}
    elif strategy == "G2_LLM_ZSA_CG":
        return {'p_distort': 5,
                'p_noise': round(random.uniform(0.7, 0.9), 2),
                'p_rotate': 3}
    elif strategy == "G3_Human_Expert":
        return {'p_distort': 5,
                'p_noise': round(random.uniform(0.8, 1.0), 2),
                'p_rotate': 25}
    elif strategy == "G4_RL_ADAPTIVE":
        return {'p_distort': 3, 'p_noise': 0.4, 'p_rotate': 15}
    else:
        return get_generation_params(random.choice(["G1_Baseline", "G2_LLM_ZSA_CG", "G3_Human_Expert"]))

def train_adaptive_model(analysis_file=DEFAULT_ANALYSIS_FILE, save_model=True, ml_model_file=ML_MODEL_FILE):
    """Train Random Forest to predict AGE from parameters."""
    if not os.path.exists(analysis_file):
        print(f"ERROR: Analysis file {analysis_file} not found.")
        return None
    df = pd.read_csv(analysis_file)
    # Rename old column if exists
    if 'strategy' in df.columns:
        df.rename(columns={'strategy': 'generation_strategy'}, inplace=True)
    required_cols = ['success','typing_time','generation_strategy','p_distort','p_noise','p_rotate']
    for col in required_cols:
        if col not in df.columns:
            print(f"ERROR: Missing column {col}")
            return None
    df.dropna(subset=required_cols, inplace=True)
    df_train = df[df['generation_strategy'].isin(['G1_Baseline','G2_LLM_ZSA_CG','G3_Human_Expert'])]
    if df_train.empty:
        print("ERROR: Training data empty")
        return None

    sr_bot = df_train[df_train['typing_time']==0].groupby(['p_distort','p_noise','p_rotate'])['success'].mean().rename('SR_bot')
    sr_human = df_train[df_train['typing_time']>0].groupby(['p_distort','p_noise','p_rotate'])['success'].mean().rename('SR_human')
    results = pd.merge(sr_bot, sr_human, left_index=True, right_index=True, how='inner')
    if results.empty:
        print("ERROR: No matching bot-human param combinations")
        return None
        
    results['AGE_Score'] = results['SR_human'] - results['SR_bot']
    
    X = results.index.to_frame(index=False).astype(float)
    Y = results['AGE_Score'].astype(float)

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X, Y)
    if save_model:
        # FIX 2: os.path.dirname will now correctly return 'models'
        os.makedirs(os.path.dirname(ml_model_file), exist_ok=True) 
        dump(rf, ml_model_file)
    return rf

def load_adaptive_model(ml_model_file=ML_MODEL_FILE):
    if os.path.exists(ml_model_file):
        try:
            return load(ml_model_file)
        except:
            return None
    return None