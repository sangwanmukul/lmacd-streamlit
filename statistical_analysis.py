import pandas as pd
from scipy.stats import ttest_ind

def run_statistical_analysis(df):
    if "generation_strategy" not in df.columns:
        print("[WARN] 'generation_strategy' column not found — assigning default groups.")
        df["generation_strategy"] = "G1"

    group1 = df[df["generation_strategy"] == "G1"]["AGE_Score"]
    group4 = df[df["generation_strategy"] == "G4"]["AGE_Score"]
    
    t_stat, p_val = ttest_ind(group1, group4, nan_policy="omit")
    return t_stat, p_val
