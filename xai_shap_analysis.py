# xai_shap_analysis.py
import shap
import pandas as pd
import matplotlib.pyplot as plt

def run_shap_analysis(df, model):
    """
    df: pandas DataFrame with features and Target
    model: trained ML model
    """
    # Extract features
    X = df.drop(columns=["Target", "generation_strategy"], errors='ignore')
    y = df["Target"]

    # SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Global summary plot
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig("plots/shap_summary_plot.png")
    plt.close()

    # SHAP importance bar
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.savefig("plots/shap_importance_bar.png")
    plt.close()

    print("[+] SHAP analysis complete")
    return shap_values
