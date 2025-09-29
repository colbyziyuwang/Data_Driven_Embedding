import numpy as np
import matplotlib.pyplot as plt
import os

# Environments to plot
envs = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1", "LunarLander-v3"]

def smooth(data, window=25):
    smoothed = np.zeros_like(data)
    for i in range(data.shape[0]):
        for t in range(data.shape[1]):
            start = max(0, t - window + 1)
            smoothed[i, t] = data[i, start:t+1].mean()
    return smoothed

def plot_all(envs, window=25):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, env in enumerate(envs):
        ax = axes[idx]
        path = f"results/{env}/"

        # Load rewards
        reward_cpae = np.load(path + "cbow.npy")       # CBOW = CPAE
        reward_sace = np.load(path + "skipgram.npy")   # SkipGram = SACE
        reward_dqn = np.load(path + "dqn.npy")

        # Apply smoothing
        reward_cpae = smooth(reward_cpae, window)
        reward_sace = smooth(reward_sace, window)
        reward_dqn = smooth(reward_dqn, window)

        x = np.arange(reward_cpae.shape[1])

        def plot_with_std(mean, std, color, label):
            lower = mean - std
            upper = mean + std
            ax.plot(x, mean, color=color, label=label)
            ax.fill_between(x, lower, upper, color=color, alpha=0.3)

        # Compute mean and std across seeds
        mean_cpae, std_cpae = np.mean(reward_cpae, axis=0), np.std(reward_cpae, axis=0)
        mean_sace, std_sace = np.mean(reward_sace, axis=0), np.std(reward_sace, axis=0)
        mean_dqn, std_dqn   = np.mean(reward_dqn, axis=0), np.std(reward_dqn, axis=0)

        # Plot each curve
        plot_with_std(mean_cpae, std_cpae, 'blue', 'CPAE')
        plot_with_std(mean_sace, std_sace, 'orange', 'SACE')
        plot_with_std(mean_dqn, std_dqn, 'green', 'DQN')

        ax.set_title(env)
        ax.set_xlabel("Episodes")
        ax.set_ylabel(f"Reward (Smoothed, {window})")
        ax.grid(True)
        if idx == 0:  # show legend only once
            ax.legend(loc="upper left")

    plt.suptitle("Reward Comparison Across Environments", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs("Results", exist_ok=True)
    plt.savefig("Results/reward_comparison_all_envs.png")
    plt.show()

# Call
plot_all(envs, window=15)
