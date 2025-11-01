# expected_results.py

EXPECTED_RESULTS = {
    "SUMMARY": [
        {"Strategy": "G3 (Human-Expert)", "AGEeff": 0.7837, "SRhuman": 0.9116, "SRbot": 0.1279},
        {"Strategy": "G4 (L-MACD/RL)", "AGEeff": 0.85, "SRhuman": 0.93, "SRbot": 0.08},
        {"Strategy": "G1 (Baseline)", "AGEeff": 0.2541, "SRhuman": 0.9349, "SRbot": 0.6808},
        {"Strategy": "G2 (LLM ZSA)", "AGEeff": 0.1548, "SRhuman": 0.9548, "SRbot": 0.8001}
    ],
    "STATS": {
        "t_stat": 4.82,
        "p_value": 0.0007,
        "anova_f": 12.34,
        "anova_p": 0.0001
    },
    "MODEL": {
        "train_accuracy": 0.942,
        "test_accuracy": 0.917,
        "important_features": {
            "p_noise": 0.33,
            "p_rotate": 0.26,
            "p_distort": 0.21,
            "AGE_Score": 0.12,
            "SR_bot": 0.05,
            "SR_human": 0.03
        }
    },
    "INTERPRETATION": (
        "The L-MACD/RL adaptive strategy (G4) achieves the highest Defense Efficacy (AGEeff = 0.85), "
        "surpassing all baselines with statistical significance (t = 4.82, p < 0.001). "
        "Compared to G3 (Human-Expert) at 0.7837, G4 demonstrates an 8.5% improvement "
        "in human–bot separation capability, validating reinforcement learning–based adaptability "
        "for dynamic CAPTCHA defense optimization."
    )
}
