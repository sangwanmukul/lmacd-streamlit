import os, random, csv
import pandas as pd
import numpy as np
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# ------------------------
# Config
# ------------------------
CAPTCHA_LENGTH = 5
ANALYSIS_FILE = "data/results_analysis.csv"
PLOTS_DIR = "plots"
MODELS_DIR = "models"
FONT_PATH = "fonts/DejaVuSans.ttf"
NUM_TRIALS = 300
TRAINING_GROUPS = ["G1_Baseline", "G2_LLM_ZSA_CG", "G3_Human_Expert"]
DEPLOYMENT_GROUP = "G4_RL_ADAPTIVE"

os.makedirs("data", exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs("fonts", exist_ok=True)

random.seed(42)
np.random.seed(42)

# ------------------------
# CAPTCHA Simulation
# ------------------------
def random_text(length=CAPTCHA_LENGTH):
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(random.choice(chars) for _ in range(length))

def get_generation_params(strategy):
    return {
        'p_distort': round(random.uniform(0.4, 0.6),2),
        'p_noise': round(random.uniform(0.4, 0.6),2),
        'p_rotate': round(random.uniform(0.4, 0.6),2)
    }

def save_metrics(metrics):
    fieldnames = ['captcha_type','captcha_text','user_input','success',
                  'typing_time','accuracy','ai_result','generation_strategy',
                  'p_distort','p_noise','p_rotate']
    file_exists = os.path.isfile(ANALYSIS_FILE)
    with open(ANALYSIS_FILE, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

def run_captcha_session(strategy, is_human_sim=False):
    captcha_text = random_text()
    params = get_generation_params(strategy)
    captcha_type_detail = random.choice(["adversarial","semantic","question"])

    human_sr, bot_sr = 0.85, 0.75
    if strategy == DEPLOYMENT_GROUP:
        human_sr, bot_sr = 0.9, 0.75

    if is_human_sim:
        success = int(random.random() < human_sr)
        typing_time = round(random.uniform(3,10),2)
        user_input = captcha_text if success else random_text()
        accuracy = 1.0 if success else 0.0
        ai_result = "HUMAN_AGENT"
    else:
        success = int(random.random() < bot_sr)
        typing_time = 0
        user_input = captcha_text if success else random_text()
        accuracy = 1.0 if success else 0.0
        ai_result = "ML_MODEL"

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
        'p_rotate': params['p_rotate']
    }
    save_metrics(metrics)

def run_simulation():
    if os.path.exists(ANALYSIS_FILE):
        os.remove(ANALYSIS_FILE)
    print(f"Running simulation → {ANALYSIS_FILE}")

    for strategy in TRAINING_GROUPS:
        for _ in range(NUM_TRIALS):
            run_captcha_session(strategy, is_human_sim=True)
            run_captcha_session(strategy, is_human_sim=False)
    for _ in range(NUM_TRIALS):
        run_captcha_session(DEPLOYMENT_GROUP, is_human_sim=True)
        run_captcha_session(DEPLOYMENT_GROUP, is_human_sim=False)

# ------------------------
# ML Model
# ------------------------
def train_adaptive_model():
    df = pd.read_csv(ANALYSIS_FILE)
    X = df[['p_distort','p_noise','p_rotate']]
    y = df['success']
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    joblib.dump(model, os.path.join(MODELS_DIR,"rf_model.pkl"))
    return model, X, y

# ------------------------
# Plots
# ------------------------
def generate_plots(df, model, X, y):
    # Confusion matrix
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(PLOTS_DIR,"confusion_matrix.png"))
    plt.close()

    # Feature importance
    plt.figure()
    sns.barplot(x=model.feature_importances_, y=X.columns)
    plt.title("Feature Importance")
    plt.savefig(os.path.join(PLOTS_DIR,"feature_importance.png"))
    plt.close()

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    shap.summary_plot(shap_values.values, X, show=False)
    plt.savefig(os.path.join(PLOTS_DIR,"shap_summary.png"))
    plt.close()

# ------------------------
# PDF Generation
# ------------------------
class PDF(FPDF):
    def header(self):
        self.set_font("DejaVu", "", 14)
        self.cell(0, 10, "L-MACD Research Full Report", ln=True, align="C")

    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "", 10)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def generate_report():
    if not os.path.exists(FONT_PATH):
        print(f"[ERROR] Font not found: {FONT_PATH}")
        return

    pdf = PDF()
    pdf.add_font("DejaVu", "", FONT_PATH, uni=True)
    pdf.add_page()
    df = pd.read_csv(ANALYSIS_FILE)

    # Summary
    pdf.set_font("DejaVu","",12)
    pdf.multi_cell(0,8, f"Total trials: {len(df)}\nGroups: {df['generation_strategy'].nunique()}")

    # Add plots
    for plot_file in os.listdir(PLOTS_DIR):
        pdf.add_page()
        pdf.image(os.path.join(PLOTS_DIR,plot_file), x=15, w=180)

    output_file = "L_MACD_Report.pdf"
    pdf.output(output_file)
    print(f"✅ Report saved → {output_file}")

# ------------------------
# Main
# ------------------------
if __name__=="__main__":
    if not os.path.exists(ANALYSIS_FILE):
        print("[INFO] Analysis file not found. Running simulation first...")
        run_simulation()
    else:
        print("[INFO] Using existing analysis file.")

    # Train model and generate plots
    model, X, y = train_adaptive_model()
    df = pd.read_csv(ANALYSIS_FILE)
    generate_plots(df, model, X, y)

    # Generate PDF
    generate_report()
