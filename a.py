#!/usr/bin/env python3
"""
real_captcha_experiment.py

Generates realistic CAPTCHA images of three types:
 - adversarial : random alpha-numeric strings with distortions/noise/rotation
 - semantic    : real words drawn from a wordlist
 - question    : simple arithmetic questions (e.g., "12+7=?")

Collects real human input (interactive) and AI input via Tesseract OCR,
saves metrics to CSV, and can train a simple RandomForest model for exploration.

Requirements:
  pip install pillow numpy matplotlib seaborn pandas pytesseract scikit-learn joblib

Make sure system tesseract is installed and available in PATH.
Place fonts into ./fonts/ (optional). Default uses PIL builtin font if none provided.
"""

import os
import sys
import csv
import time
import random
import math
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import pytesseract  # make sure tesseract is installed on system
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

# ------------------ CONFIG ------------------
DATA_DIR = Path("data")
PLOTS_DIR = Path("plots")
FONTS_DIR = Path("fonts")
OUT_CSV = DATA_DIR / "real_captcha_results.csv"
MODEL_DIR = Path("models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(FONTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Experiment settings
NUM_TRIALS_PER_GROUP = 30   # adjust for real human experiments (small for pilot)
CAPTCHA_LENGTH = 5
IMAGE_SIZE = (260, 90)
FONT_SIZE = 48
HANDWRITTEN_FONT = FONTS_DIR / "Handwritten.ttf"    # optional
REGULAR_FONT = FONTS_DIR / "DejaVuSans.ttf"         # optional

# groups (you can map to your G1..G4)
TRAINING_GROUPS = ["G1_Baseline", "G2_LLM_ZSA_CG", "G3_Human_Expert"]
DEPLOYMENT_GROUP = "G4_RL_ADAPTIVE"

# Human collection toggle (interactive). If False, only AI solver runs.
COLLECT_HUMAN = True

# AI solver toggle: use pytesseract if True (requires tesseract installed)
USE_TESSERACT = True

# Seeds for reproducibility
RND_SEED = 42
random.seed(RND_SEED)
np.random.seed(RND_SEED)

# Wordlist for semantic CAPTCHAS (you can expand this)
WORD_LIST = [
    "APPLE","BANANA","ORANGE","MANGO","GRAPE","PEAR","LEMON","KIWI","PEACH","PLUM",
    "RIVER","MOUNTAIN","MUSIC","COMPUTER","PYTHON","SCIENCE","ROBOT","DRAGON","GALAXY","BRIDGE"
]

# ------------------ UTILITIES ------------------
def load_font(font_path: Path, size: int):
    """Try to load a TTF font; fall back to default PIL font."""
    try:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
        else:
            return ImageFont.truetype(str(REGULAR_FONT), size=size) if REGULAR_FONT.exists() else ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()

def add_noise(img: Image.Image, amount=0.05):
    """Add random Gaussian noise to image (amount: fraction of pixels)."""
    arr = np.array(img).astype(np.int16)
    noise = np.random.normal(0, 25, arr.shape).astype(np.int16)
    arr = arr + (noise * amount)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def distort_image(img: Image.Image, max_shear=0.2):
    """Apply a simple shear/distortion."""
    width, height = img.size
    m = max_shear * (random.random()*2 - 1)
    xshift = abs(m) * width
    new_width = width + int(round(xshift))
    img = img.transform((new_width, height), Image.AFFINE,
                        (1, m, -xshift if m > 0 else 0, 0, 1, 0),
                        Image.BICUBIC)
    return img

def random_rotation(img: Image.Image, max_degrees=25):
    angle = random.uniform(-max_degrees, max_degrees)
    return img.rotate(angle, Image.BILINEAR, expand=1, fillcolor=(255,255,255))

def add_lines(img: Image.Image, line_count=3):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for _ in range(line_count):
        x1, y1 = random.randint(0, w//2), random.randint(0, h)
        x2, y2 = random.randint(w//2, w), random.randint(0, h)
        draw.line(((x1,y1),(x2,y2)), fill=(0,0,0), width=random.randint(1,3))
    return img

def crop_or_resize_to_target(img, target_size=IMAGE_SIZE):
    # if bigger, center-crop; if smaller, pad with white
    img = img.convert("RGB")
    w, h = img.size
    tw, th = target_size
    if w > tw or h > th:
        left = max(0, (w - tw)//2)
        top = max(0, (h - th)//2)
        img = img.crop((left, top, left+tw, top+th))
    elif w < tw or h < th:
        new = Image.new("RGB", target_size, (255,255,255))
        new.paste(img, ((tw-w)//2, (th-h)//2))
        img = new
    return img

def preprocess_for_ocr(img: Image.Image):
    """Preprocessing steps before pytesseract: grayscale, threshold, maybe blur."""
    gray = img.convert("L")
    # increase contrast
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(1.5)
    # adaptive threshold via numpy
    arr = np.array(gray)
    # simple global Otsu-like threshold
    thresh = arr.mean()
    bin_arr = (arr > thresh).astype(np.uint8) * 255
    bin_img = Image.fromarray(bin_arr)
    # optional morphological blur
    bin_img = bin_img.filter(ImageFilter.MedianFilter(size=3))
    return bin_img

# ------------------ CAPTCHA GENERATION ------------------
def generate_captcha_text(captcha_type: str):
    if captcha_type == "adversarial":
        charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return "".join(random.choice(charset) for _ in range(CAPTCHA_LENGTH))
    elif captcha_type == "semantic":
        # pick a real word (could be longer/shorter; we may pad or resize font)
        return random.choice(WORD_LIST)
    elif captcha_type == "question":
        # simple arithmetic
        a = random.randint(2, 20)
        b = random.randint(1, 20)
        op = random.choice(["+","-","*"])
        if op == "+":
            ans = a + b
        elif op == "-":
            ans = a - b
        else:
            ans = a * b
        # store question as "a op b = ?" and answer separately when saving metrics
        return f"{a}{op}{b}=?"
    else:
        raise ValueError("Unknown captcha_type")

def render_captcha_image(text: str, captcha_type: str,
                         p_distort=0.4, p_noise=0.4, p_rotate=12,
                         use_handwritten=False):
    """
    Render text to image, apply distortions/noise/rotation.
    p_distort, p_noise: 0..1 controlling intensity
    p_rotate: max rotation degrees
    """
    # choose font
    font = load_font(HANDWRITTEN_FONT if use_handwritten and HANDWRITTEN_FONT.exists() else REGULAR_FONT, FONT_SIZE)
    # initial large canvas to allow rotation without clipping
    canv_w, canv_h = IMAGE_SIZE[0]*2, IMAGE_SIZE[1]*2
    img = Image.new("RGB", (canv_w, canv_h), (255,255,255))
    draw = ImageDraw.Draw(img)

    # Center text
    font_to_use = font
    if len(text) > 8:
        for fs in [FONT_SIZE, FONT_SIZE-8, FONT_SIZE-12, FONT_SIZE-16]:
            font_try = load_font(REGULAR_FONT, fs)
            bbox = draw.textbbox((0,0), text, font=font_try)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if w < canv_w*0.9:
                font_to_use = font_try
                break

    bbox = draw.textbbox((0,0), text, font=font_to_use)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pos = ((canv_w - w)//2 + random.randint(-10,10), (canv_h - h)//2 + random.randint(-6,6))

    # draw baseline text
    text_color = (0,0,0)
    draw.text(pos, text, font=font_to_use, fill=text_color)

    # optionally overlay small dots for adversarial
    if captcha_type == "adversarial":
        for _ in range(int(50 * p_noise)):
            x = random.randint(0, canv_w-1)
            y = random.randint(0, canv_h-1)
            draw.point((x,y), fill=(random.randint(0,150),)*3)

    # distort & shear
    if p_distort > 0.05:
        img = distort_image(img, max_shear=0.15 * p_distort)

    # rotate randomly
    img = random_rotation(img, max_degrees=p_rotate)

    # crop/resize to final IMAGE_SIZE
    img = crop_or_resize_to_target(img, IMAGE_SIZE)

    # add lines for adversarial
    if captcha_type == "adversarial":
        img = add_lines(img, line_count=int(3 * p_distort))

    # noise
    if p_noise > 0.05:
        img = add_noise(img, amount=p_noise*0.8)

    # final slight blur/sharpen
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.2)))
    else:
        img = ImageEnhance.Sharpness(img).enhance(1.0 + (p_distort*0.5))

    return img

# ------------------ HUMAN / AI COLLECTION ------------------
def collect_human_input_interactive(img: Image.Image, correct_answer: str):
    """Show image to human and collect typed response + typing time."""
    # Save temp file and open (Image.show uses default viewer)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        img.save(tmp.name)
        tmp.flush()
        # open image
        print(f"[HUMAN] Opening image for human input: {tmp.name}")
        Image.open(tmp.name).show()
        start = time.time()
        user_input = input("Enter CAPTCHA text (press Enter when done): ").strip()
        elapsed = time.time() - start
        # normalize
        success = 0
        user_input_norm = user_input.strip()
        # For question, evaluate expected numeric answer
        if "?" in correct_answer and "=" in correct_answer:
            # question form: e.g., '12+7=?'
            try:
                expr = correct_answer.split('=')[0]
                expected = str(eval(expr))
                success = int(user_input_norm == expected)
            except Exception:
                success = 0
        else:
            success = int(user_input_norm.upper() == correct_answer.upper())
        # Close image viewer: can't programmatically close on all systems; user will close manually.
        return user_input_norm, elapsed, success, 1.0 if success else 0.0
    finally:
        try:
            tmp.close()
        except Exception:
            pass

def collect_ai_input_tesseract(img: Image.Image, correct_answer: str):
    """Run pytesseract OCR on the input image (after preprocessing)."""
    if not USE_TESSERACT:
        return "", 0.0, 0, 0.0

    # Preprocess
    proc = preprocess_for_ocr(img)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        proc.save(tmp.name)
        # Call tesseract
        try:
            raw = pytesseract.image_to_string(Image.open(tmp.name), config='--psm 7')  # single line
            pred = raw.strip().replace(" ", "").upper()
        except Exception as e:
            print("[AI] Tesseract error:", e)
            pred = ""
    finally:
        try:
            tmp.close()
        except Exception:
            pass

    # Evaluate success
    if "?" in correct_answer and "=" in correct_answer:
        try:
            expr = correct_answer.split('=')[0]
            expected = str(eval(expr))
            success = int(pred == expected)
        except Exception:
            success = 0
    else:
        success = int(pred == correct_answer.upper())

    return pred, 0.0, success, 1.0 if success else 0.0

# ------------------ METRICS AND LOGGING ------------------
CSV_FIELDS = [
    "timestamp", "group", "captcha_type", "captcha_text", "rendered_text",
    "p_distort", "p_noise", "p_rotate",
    "agent_type", "agent_response", "success", "typing_time", "accuracy"
]

def append_row_to_csv(row: dict):
    exists = OUT_CSV.exists()
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

# ------------------ MAIN EXPERIMENT LOOP ------------------
def run_experiment(num_trials_per_group=NUM_TRIALS_PER_GROUP, collect_human=COLLECT_HUMAN):
    groups = TRAINING_GROUPS + [DEPLOYMENT_GROUP]
    # param ranges: you can change to use trained RL / ML model values
    for group in groups:
        print(f"\n=== Running trials for {group} ===")
        for t in range(num_trials_per_group):
            # pick a random captcha type for diversity or you can iterate
            captcha_type = random.choice(["adversarial", "semantic", "question"])
            # set generation params according to group if desired
            if group == "G1_Baseline":
                p_distort = round(random.uniform(0.2, 0.4),2)
                p_noise = round(random.uniform(0.2, 0.4),2)
                p_rotate = random.uniform(5, 12)
            elif group == "G2_LLM_ZSA_CG":
                p_distort = round(random.uniform(0.5, 0.8),2)
                p_noise = round(random.uniform(0.5, 0.9),2)
                p_rotate = random.uniform(0, 6)
            elif group == "G3_Human_Expert":
                p_distort = round(random.uniform(0.6, 0.9),2)
                p_noise = round(random.uniform(0.6, 1.0),2)
                p_rotate = random.uniform(15, 30)
            else:  # G4 RL adaptive (use lower AGE heuristics)
                p_distort = round(random.uniform(0.1, 0.35),2)
                p_noise = round(random.uniform(0.1, 0.35),2)
                p_rotate = random.uniform(8, 20)

            captcha_text = generate_captcha_text(captcha_type)
            # If question type, captcha_text contains 'a+b=?'; keep as the text
            img = render_captcha_image(captcha_text, captcha_type,
                                       p_distort=p_distort, p_noise=p_noise, p_rotate=p_rotate,
                                       use_handwritten=(captcha_type=="semantic" and HANDWRITTEN_FONT.exists()))

            # Save sample image (optional)
            stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
            fname = DATA_DIR / f"{group}_{captcha_type}_{t}_{stamp}.png"
            img.save(fname)

            # HUMAN COLLECTION
            if collect_human and captcha_type in ("semantic", "adversarial", "question"):
                print(f"\n[Trial {t+1}/{num_trials_per_group}] Group={group}, Type={captcha_type}, Image={fname}")
                human_resp, typing_time, success_h, acc_h = collect_human_input_interactive(img, captcha_text)
                row = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "group": group,
                    "captcha_type": captcha_type,
                    "captcha_text": captcha_text,
                    "rendered_text": fname.name,
                    "p_distort": p_distort,
                    "p_noise": p_noise,
                    "p_rotate": p_rotate,
                    "agent_type": "HUMAN",
                    "agent_response": human_resp,
                    "success": success_h,
                    "typing_time": round(typing_time, 3),
                    "accuracy": acc_h
                }
                append_row_to_csv(row)
            else:
                print(f"[Skipping human] Group={group}, Type={captcha_type}, Image={fname}")

            # AI COLLECTION (Tesseract)
            if USE_TESSERACT:
                ai_pred, ai_time, success_ai, acc_ai = collect_ai_input_tesseract(img, captcha_text)
                row_ai = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "group": group,
                    "captcha_type": captcha_type,
                    "captcha_text": captcha_text,
                    "rendered_text": fname.name,
                    "p_distort": p_distort,
                    "p_noise": p_noise,
                    "p_rotate": p_rotate,
                    "agent_type": "TESSERACT",
                    "agent_response": ai_pred,
                    "success": success_ai,
                    "typing_time": round(ai_time, 3),
                    "accuracy": acc_ai
                }
                append_row_to_csv(row_ai)
                print(f"[AI] pred='{ai_pred}' success={success_ai}")

    print("\n=== Experiment complete ===")
    print(f"Results saved to {OUT_CSV}")

# ------------------ SIMPLE ANALYSIS / PLOTS ------------------
def run_basic_analysis_and_plots():
    if not OUT_CSV.exists():
        print("[WARN] No CSV found to analyze.")
        return
    df = pd.read_csv(OUT_CSV)
    print(f"Loaded {len(df)} records")

    # success rate by group & agent type
    summary = df.groupby(["group", "agent_type"])["success"].agg(["count","mean"]).reset_index()
    print(summary)

    # save barplot
    plt.figure(figsize=(8,4))
    sns.barplot(data=summary, x="group", y="mean", hue="agent_type")
    plt.ylabel("Mean success rate")
    plt.title("Success rate by group and agent")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "success_rate_group_agent.png")
    plt.close()

    # confusion / accuracy for AI (requires truth = captcha_text vs agent_response)
    ai_df = df[df["agent_type"]=="TESSERACT"].copy()
    if not ai_df.empty:
        ai_df["match"] = ai_df["success"]
        acc = ai_df["match"].mean()
        print(f"Tesseract accuracy overall: {acc:.3f}")

    # basic feature importance: train RF to predict success using p_distort, p_noise, p_rotate
    numeric = df[["p_distort","p_noise","p_rotate","success"]].dropna()
    if len(numeric) > 50:
        X = numeric[["p_distort","p_noise","p_rotate"]]
        y = numeric["success"]
        clf = RandomForestClassifier(n_estimators=100, random_state=RND_SEED)
        clf.fit(X, y)
        importances = clf.feature_importances_
        plt.figure()
        sns.barplot(x=importances, y=X.columns)
        plt.title("Feature importance for predicting success")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "feature_importance.png")
        plt.close()
        joblib.dump(clf, MODEL_DIR / "rf_success_predictor.pkl")
        print("Saved RF model for success prediction.")
    else:
        print("Not enough numeric records for feature importance training.")

    print(f"Saved plots to {PLOTS_DIR}")

# ------------------ ENTRYPOINT ------------------
if __name__ == "__main__":
    print("=== Real CAPTCHA Experiment Launcher ===")
    print("Make sure you have Tesseract installed on your system for AI OCR.")
    print(f"Output CSV: {OUT_CSV}")
    print(f"Fonts dir (optional): {FONTS_DIR}")

    # Ask user whether to run interactive human collection
    if COLLECT_HUMAN:
        print("\n[CONFIG] Human collection ENABLED (interactive).")
        print("To skip human collection and run AI only, set COLLECT_HUMAN=False in the script.")
    else:
        print("\n[CONFIG] Human collection DISABLED — will run AI only.")

    try:
        run_experiment(num_trials_per_group=NUM_TRIALS_PER_GROUP, collect_human=COLLECT_HUMAN)
        run_basic_analysis_and_plots()
        print("\nAll done. Inspect CSV and plots for results.")
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
        sys.exit(0)
