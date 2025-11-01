# rl_agent.py
import numpy as np

def simulate_reward(candidate):
    """
    Dummy reward function: higher sum of candidate vector gives higher reward.
    Replace with your real evaluation logic.
    """
    return np.tanh(np.sum(candidate))  # example: reward between -1 and 1

def hillclimb_optimize(n_steps=50, n_params=3):
    """
    Hill Climbing RL optimization
    Returns: best_action, best_reward, final_epsilon
    """
    best_action = np.random.rand(n_params)
    best_reward = simulate_reward(best_action)
    epsilon = 0.1

    for step in range(n_steps):
        candidate = best_action + np.random.normal(0, 0.05, size=n_params)
        reward = simulate_reward(candidate)
        if reward > best_reward:
            best_reward = reward
            best_action = candidate
        epsilon = max(0.01, epsilon * 0.99)

    return best_action, best_reward, epsilon
