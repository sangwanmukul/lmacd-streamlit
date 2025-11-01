# visualization.py (FINAL COMPLETE CODE)
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve # Import for stability analysis

# === Utility: Safe plot saving ===
def save_plot(filename: str):
    """Save current matplotlib plot safely."""
    # Ensure 'plots' directory exists before saving
    os.makedirs("plots", exist_ok=True) 
    path = os.path.join("plots", filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved plot → {path}")

# === Model Accuracy Trend (Learning Curve - The large dataset plot) ===
# THIS FUNCTION RESOLVES THE ImportError
def plot_learning_curve(model, X, y, filename="model_learning_curve.png"):
    """Generates a learning curve to show model stability and generalization."""
    
    # Generate cross-validation statistics for a learning curve
    # train_sizes are samples of the AGGREGATED data 
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.grid()

    plt.plot(train_sizes, train_scores_mean, 'o-', color="red", label="Training Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="blue", label="Cross-validation Score")

    plt.title("Model Stability and Generalization (Learning Curve)")
    plt.xlabel("Training Examples (Unique Configurations)")
    plt.ylabel("Score (Accuracy)")
    plt.legend(loc="best")
    save_plot(filename)


# === Confusion Matrix ===
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap="Blues", values_format="d", ax=ax) 
    plt.title(title)
    save_plot("confusion_matrix.png")

# === Feature Importance ===
def plot_feature_importance(importances, feature_names):
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(6, 4))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    save_plot("feature_importance.png")