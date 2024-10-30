import numpy as np
import matplotlib.pyplot as plt

# Simulate the loading of CBOW and SkipGram rewards for the example (since I cannot access files)
reward_cbow = np.load("reward_cbow.npy")
reward_skipgram = np.load("reward_skipgram.npy")

# Function to plot two curves
def plot_two_curves(reward_cbow, reward_skipgram):
    plt.figure(figsize=(10, 6))

    # Calculate mean and std for both CBOW and SkipGram
    mean_cbow = np.mean(reward_cbow, axis=0)
    std_cbow = np.std(reward_cbow, axis=0)

    mean_skipgram = np.mean(reward_skipgram, axis=0)
    std_skipgram = np.std(reward_skipgram, axis=0)

    # Plot CBOW curve
    plt.plot(range(len(mean_cbow)), mean_cbow, color='blue', label='CBOW Reward')
    plt.fill_between(range(len(mean_cbow)), np.maximum(mean_cbow - std_cbow, 0), np.minimum(mean_cbow + std_cbow, 200), color='blue', alpha=0.3)

    # Plot SkipGram curve
    plt.plot(range(len(mean_skipgram)), mean_skipgram, color='orange', label='SkipGram Reward')
    plt.fill_between(range(len(mean_skipgram)), np.maximum(mean_skipgram - std_skipgram, 0), np.minimum(mean_skipgram + std_skipgram, 200), color='orange', alpha=0.3)

    # Add labels and title
    plt.xlabel("Episodes")  # Label for x-axis
    plt.ylabel("Average Reward (Last 25 Test Episodes)")  # Label for y-axis
    plt.title("Comparison of Average Rewards on Test Env for CBOW and SkipGram Models")  # Title of the plot
    plt.legend(loc="upper left")  # Legend location
    plt.grid(True)  # Add grid for better readability

    # Save the plot as a PNG file
    plt.savefig("reward_comparison.png")

# Call the function to generate and save the plot
plot_two_curves(reward_cbow, reward_skipgram)
