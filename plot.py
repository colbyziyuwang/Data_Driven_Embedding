import numpy as np
import matplotlib.pyplot as plt
import os

env = "LunarLander-v3" # "CartPole-v1", "MountainCar-v0", "Acrobot-v1", "LunarLander-v3"
path = f"results/{env}/"

# Load rewards
reward_cbow = np.load(path + "cbow.npy")
reward_skipgram = np.load(path + "skipgram.npy")
reward_dqn = np.load(path + "dqn.npy")

def smooth(data, window=25):
    smoothed = np.zeros_like(data)
    for i in range(data.shape[0]):
        for t in range(data.shape[1]):
            start = max(0, t - window + 1)
            smoothed[i, t] = data[i, start:t+1].mean()
    return smoothed

def plot_curves(reward_cbow, reward_skipgram, reward_dqn, window=25):
    plt.figure(figsize=(12, 6))
    
    # Apply smoothing
    reward_cbow = smooth(reward_cbow, window)
    reward_skipgram = smooth(reward_skipgram, window)
    reward_dqn = smooth(reward_dqn, window)
    
    x = np.arange(reward_cbow.shape[1])  # Assume shape: (num_seeds, num_episodes)
    
    def plot_with_std(mean, std, color, label):
        lower = mean - std
        upper = mean + std
        plt.plot(x, mean, color=color, label=label)
        plt.fill_between(x, lower, upper, color=color, alpha=0.3)

    # Compute mean and std across seeds
    mean_cbow = np.mean(reward_cbow, axis=0)
    std_cbow = np.std(reward_cbow, axis=0)
    
    mean_skipgram = np.mean(reward_skipgram, axis=0)
    std_skipgram = np.std(reward_skipgram, axis=0)
    
    mean_dqn = np.mean(reward_dqn, axis=0)
    std_dqn = np.std(reward_dqn, axis=0)

    # Plot each curve
    plot_with_std(mean_cbow, std_cbow, 'blue', 'CBOW')
    plot_with_std(mean_skipgram, std_skipgram, 'orange', 'SkipGram')
    plot_with_std(mean_dqn, std_dqn, 'green', 'DQN')

    plt.xlabel("Episodes")
    plt.ylabel("Average Reward (Smoothed, Window = {})".format(window))
    plt.title(f"Reward Comparison on {env}")
    plt.grid(True)
    plt.legend(loc="upper left")

    # Save and show
    os.makedirs("Results", exist_ok=True)
    plt.savefig(f"Results/reward_comparison_{env}.png")
    plt.show()

# Call with smoothing window
plot_curves(reward_cbow, reward_skipgram, reward_dqn, window=15)
