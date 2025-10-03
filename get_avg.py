import numpy as np

envs = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1", "LunarLander-v3"]
methods = {"dqn": "DQN", "cbow": "CPAE", "skipgram": "SACE"}

# Config
LAST_N = 20  # number of episodes to average at the end

for env in envs:
    print(f"\n--- {env} ---")
    for method_key, method_name in methods.items():
        data = np.load(f"results/{env}/{method_key}.npy")  
        # data shape: (num_seeds, num_episodes)

        # (1) Final episode only
        final_rewards = data[:, -1]

        # (2) Last-N average
        lastN_rewards = data[:, -LAST_N:].mean(axis=1)

        # (3) Mean across all episodes (AUC style)
        mean_rewards = data.mean(axis=1)

        # Print nicely
        print(f"{method_name}:")
        print(f"  Final:     ${np.mean(final_rewards):.2f} \\pm {np.std(final_rewards):.2f}$")
        print(f"  Last {LAST_N}: ${np.mean(lastN_rewards):.2f} \\pm {np.std(lastN_rewards):.2f}$")
        print(f"  Mean (AUC): ${np.mean(mean_rewards):.2f} \\pm {np.std(mean_rewards):.2f}$")
