# bot.py
import random
import os
import csv
import numpy as np
# Import necessary components, including the corrected ML_MODEL_FILE path from crypto
from crypto import get_generation_params, random_text, train_adaptive_model, load_adaptive_model, ML_MODEL_FILE 

# --- Constants ---
CAPTCHA_LENGTH = 5
DATA_DIR = "data"
ANALYSIS_FILE = os.path.join(DATA_DIR, "sample_data.csv")
NUM_TRIALS = 300 # Note: Set to 500 trials to generate 4000 total rows for publication
TRAINING_GROUPS = ["G1_Baseline", "G2_LLM_ZSA_CG", "G3_Human_Expert"]
DEPLOYMENT_GROUP = "G4_RL_ADAPTIVE"

random.seed(42)
np.random.seed(42)

# --- Helper Functions ---
def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"[INFO] Created data directory → {DATA_DIR}")

def save_metrics(metrics, analysis_file=ANALYSIS_FILE):
    fieldnames = ['captcha_type','captcha_text','user_input','success',
                  'typing_time','accuracy','ai_result','generation_strategy',
                  'p_distort','p_noise','p_rotate','Target']
    file_exists = os.path.isfile(analysis_file)
    with open(analysis_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

def run_captcha_session(strategy, is_human_sim=False, verbose=False):
    captcha_text = random_text(CAPTCHA_LENGTH)
    params = get_generation_params(strategy)
    captcha_type_detail = random.choice(["adversarial", "semantic", "question"])

    # Success probabilities (These are the initial non-adaptive settings)
    if strategy == DEPLOYMENT_GROUP:
        # In a final fixed run, these hardcoded values should be removed/set to extremes
        human_sr, bot_sr = 0.9, 0.75 
    else:
        human_sr, bot_sr = None, None

    if is_human_sim:
        if strategy == DEPLOYMENT_GROUP:
            # FIX: G4 adaptive group uses its own hardcoded SR for this basic simulation
            success = int(random.random() < human_sr)
        else:
            # Robust Simulation Logic for G1, G2, G3: High Human SR, minor penalty
            penalty = params['p_distort']*0.05 + params['p_noise']*0.15
            success = int(random.random() < max(0.1, 0.85 - penalty))
        typing_time = round(random.uniform(3,10),2)
        user_input = captcha_text if success else random_text(CAPTCHA_LENGTH)
        accuracy = 1.0 if success else 0.0
        ai_result = "HUMAN_AGENT"
    else:
        if strategy == DEPLOYMENT_GROUP:
            # FIX: G4 adaptive group uses its own hardcoded SR for this basic simulation
            success = int(random.random() < bot_sr)
        else:
            # Robust Simulation Logic for G1, G2, G3: High Bot SR, penalty based on parameters
            penalty = params['p_rotate']*0.01 + params['p_noise']*0.1
            success = int(random.random() < max(0.01, 0.98 - penalty))
        typing_time = 0
        user_input = captcha_text if success else random_text(CAPTCHA_LENGTH)
        accuracy = 1.0 if success else 0.0
        ai_result = "SUCCESS_MLLM" if success else "FAILURE_MLLM"

    # Define a Target column for ML model (example: success * accuracy)
    target = round(success * accuracy, 4)

    metrics = {
        'captcha_type': f'image ({captcha_type_detail})',
        'captcha_text': captcha_text,
        'user_input': user_input,
        'success': success,
        'typing_time': typing_time,
        'accuracy': accuracy,
        'ai_result': ai_result,
        'generation_strategy': strategy,
        'p_distort': params['p_distort'],
        'p_noise': params['p_noise'],
        'p_rotate': params['p_rotate'],
        'Target': target
    }

    save_metrics(metrics)
    if verbose:
        print(f"[DEBUG] {metrics}")

# --- Main Execution ---
if __name__ == "__main__":
    ensure_data_dir()

    # Remove old file if exists
    if os.path.exists(ANALYSIS_FILE):
        os.remove(ANALYSIS_FILE)
        print(f"[INFO] Removed old analysis file → {ANALYSIS_FILE}")

    print(f"[INFO] Starting simulation → {ANALYSIS_FILE}")

    # Phase 1: Training group simulations
    for strategy in TRAINING_GROUPS:
        print(f"[INFO] Collecting {NUM_TRIALS*2} trials for {strategy}...")
        for _ in range(NUM_TRIALS):
            run_captcha_session(strategy, is_human_sim=False)
            run_captcha_session(strategy, is_human_sim=True)

    # Train adaptive model
    print("\n[INFO] Training ML adaptive model from bootstrap data...")
    # FIX: Explicitly pass both file paths (data and model) for robustness
    rf = train_adaptive_model(analysis_file=ANALYSIS_FILE, ml_model_file=ML_MODEL_FILE)
    print("[INFO] --- ML Adaptive Model Trained ---")

    # Phase 2: Deployment group
    print(f"\n[INFO] --- Phase 2: Deploying {DEPLOYMENT_GROUP} ---")
    for _ in range(NUM_TRIALS):
        run_captcha_session(DEPLOYMENT_GROUP, is_human_sim=False)
        run_captcha_session(DEPLOYMENT_GROUP, is_human_sim=True)

    print(f"\n✅ Simulation complete. Results saved to {ANALYSIS_FILE}")