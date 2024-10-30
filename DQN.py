import gym
import numpy as np
import utils.envs, utils.seed, utils.buffers, utils.torch, utils.common
import torch
import tqdm
import matplotlib.pyplot as plt
import warnings
from embedding_cbow import StateActionPredictionModel
from embedding_skipgram import SkipGramActionPredictionModel
from load_data import StateActionDataset, SkipGramStateActionDataset
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore")

# Deep Q Learning

# Constants
SEEDS = [1,2,3,4,5]
t = utils.torch.TorchHelper()
DEVICE = t.device
OBS_N = 4               # State space size
ACT_N = 2               # Action space size
MINIBATCH_SIZE = 10     # How many examples to sample per train step
GAMMA = 0.99            # Discount factor in episodic reward objective
LEARNING_RATE = 5e-4    # Learning rate for Adam optimizer
TRAIN_AFTER_EPISODES = 10   # Just collect episodes for these many episodes
TRAIN_EPOCHS = 5        # Train for these many epochs every time
BUFSIZE = 10000         # Replay buffer size
EPISODES = 300          # Total number of episodes to learn over
TEST_EPISODES = 1       # Test episodes after every train episode
HIDDEN = 512            # Hidden nodes
TARGET_UPDATE_FREQ = 10 # Target network update frequency
STARTING_EPSILON = 1.0  # Starting epsilon
STEPS_MAX = 10000       # Gradually reduce epsilon over these many steps
EPSILON_END = 0.01      # At the end, keep epsilon at this value

# Global variables
EPSILON = STARTING_EPSILON
Q = None

# Create environment
# Create replay buffer
# Create network for Q(s, a)
# Create target network
# Create optimizer
# Create embedding model (cbow or skipgram)
def create_everything(seed, cbow_flag):

    utils.seed.seed(seed)
    env = gym.make("CartPole-v0")
    env.reset(seed=seed)
    test_env = gym.make("CartPole-v0")
    test_env.reset(seed=10 + seed)
    buf = utils.buffers.ReplayBuffer(BUFSIZE)
    Q = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N)
    ).to(DEVICE)
    Qt = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N)
    ).to(DEVICE)
    OPT = torch.optim.Adam(Q.parameters(), lr = LEARNING_RATE)

    if (cbow_flag): # decide the model
        embed_model = StateActionPredictionModel(OBS_N, ACT_N)
    else:
        embed_model = SkipGramActionPredictionModel(OBS_N, ACT_N)
    
    return env, test_env, buf, Q, Qt, OPT, embed_model

# Update a target network using a source network
def update(target, source):
    for tp, p in zip(target.parameters(), source.parameters()):
        tp.data.copy_(p.data)

# Create epsilon-greedy policy
def policy(env, obs):

    global EPSILON, Q
    
    if (isinstance(obs, tuple)):
        obs = t.f(obs[0]).view(-1, OBS_N)  # Convert to torch tensor
    else:
        obs = t.f(obs).view(-1, OBS_N)  # Convert to torch tensor
    
    # With probability EPSILON, choose a random action
    # Rest of the time, choose argmax_a Q(s, a) 
    if np.random.rand() < EPSILON:
        action = np.random.randint(ACT_N)
    else:
        qvalues = Q(obs)
        action = torch.argmax(qvalues).item()
    
    # Epsilon update rule: Keep reducing a small amount over
    # STEPS_MAX number of steps, and at the end, fix to EPSILON_END
    EPSILON = max(EPSILON_END, EPSILON - (1.0 / STEPS_MAX))
    # print(EPSILON)

    return action


# Update networks
def update_networks(epi, buf, Q, Qt, OPT, embed_cbow):
    
    # Sample a minibatch (s, a, r, s', d)
    # Each variable is a vector of corresponding values
    S, A, R, S2, D = buf.sample(MINIBATCH_SIZE, t)
    
    # Get Q(s, a) for every (s, a) in the minibatch
    qvalues = Q(S).gather(1, A.view(-1, 1)).squeeze()

    # Get max_a' Qt(s', a') for every (s') in the minibatch
    # q2values = torch.max(Qt(S2), dim = 1).values

    # Get Q(s', a') for every next state s'
    q2values_all = Qt(S2)  # This gives Q-values for all actions for each state s'

    # Compute action weights using the distances
    # Assuming compute_distance already returns the weights (softmax applied internally)
    weights = embed_cbow.compute_distance(S2)  # shape [batch_size, n_actions]

    # Use the computed weights to get the weighted combination of Q-values
    q2values = torch.sum(weights * q2values_all, dim=1)

    # If done, 
    #   y = r(s, a) + GAMMA * max_a' Q(s', a') * (0)
    # If not done,
    #   y = r(s, a) + GAMMA * max_a' Q(s', a') * (1)       
    targets = R + GAMMA * q2values * (1-D)

    # Detach y since it is the target. Target values should
    # be kept fixed.
    loss = torch.nn.MSELoss()(targets.detach(), qvalues)

    # Backpropagation
    OPT.zero_grad()
    loss.backward()
    OPT.step()

    # Update target network every few steps
    if epi % TARGET_UPDATE_FREQ == 0:
        update(Qt, Q)

    return loss.item()

def create_batch_data(state_action_sequences, cbow=True):
    """
    Processes the input sequence of state-action pairs and creates a DataLoader for training.
    
    Args:
        state_action_sequences (list): A list where each element is a sequence of the form [s1, a1, s2, a2, ..., sn]
        cbow (bool): If True, processes data for CBOW; if False, processes for SkipGram.
    
    Returns:
        DataLoader: A PyTorch DataLoader for training batches.
    """
    
    # Initialize processed list
    processed_state_action_sequences = [] 
    
    if cbow:
        # CBOW Version: Predict missing action (a2) from context (s1, a1, s2, s3, a3)
        for seq in state_action_sequences:
            for i in range(0, len(seq) - 6, 2):  # Iterate over the sequence, step by 2
                s1, a1 = seq[i], seq[i+1]
                s2, a2 = seq[i+2], seq[i+3]
                s3, a3 = seq[i+4], seq[i+5]
                
                # Append the tuple (s1, a1, s2, s3, a3) and the target action a2
                processed_state_action_sequences.append((s1, a1, s2, a2, s3, a3))
        
        # Create the dataset using CBOW logic
        dataset = StateActionDataset(processed_state_action_sequences)
    
    else:
        # SkipGram Version: Predict actions (previous and next) from the current state
        for seq in state_action_sequences:
            for i in range(1, len(seq) - 2, 2):  # Step by 2 to handle action-centered tuples
                a1, s2, a2 = seq[i], seq[i+1], seq[i+2]
                
                # Append the tuple (a1, s2, a2) to the processed list
                processed_state_action_sequences.append((a1, s2, a2))
        
        # Create the dataset using SkipGram logic
        dataset = SkipGramStateActionDataset(processed_state_action_sequences)

    # Initialize DataLoader
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    return data_loader

# Play episodes
# Training function
def train(seed, cbow_flag):

    global EPSILON, Q
    print("Seed=%d" % seed)

    # Create environment, buffer, Q, Q target, optimizer
    env, test_env, buf, Q, Qt, OPT, embed_model = create_everything(seed, cbow_flag)
    state_action_sequences = []

    # epsilon greedy exploration
    EPSILON = STARTING_EPSILON

    testRs = []
    last25testRs = []
    print("Training:")
    pbar = tqdm.trange(EPISODES)
    for epi in pbar:

        # Play an episode and log episodic reward (also collects state-action sequences)
        S, A, R, s_a_seq = utils.envs.play_episode_rb(env, policy, buf)
        state_action_sequences.append(s_a_seq)
        
        # Train after collecting sufficient experience
        if epi >= TRAIN_AFTER_EPISODES:

            # Train for TRAIN_EPOCHS
            for tri in range(TRAIN_EPOCHS):
                data_loader = create_batch_data(state_action_sequences, cbow_flag)
                embed_model.train(data_loader)
                update_networks(epi, buf, Q, Qt, OPT, embed_model)
            state_action_sequences = []

        # Evaluate for TEST_EPISODES number of episodes
        Rews = []
        for epj in range(TEST_EPISODES):
            S, A, R = utils.envs.play_episode(test_env, policy, render = False)
            Rews += [sum(R)]
        testRs += [sum(Rews)/TEST_EPISODES]

        # Update progress bar
        last25testRs += [sum(testRs[-25:])/len(testRs[-25:])]
        pbar.set_description("R25(%g)" % (last25testRs[-1]))

    # Close progress bar, environment
    pbar.close()
    print("Training finished!")
    env.close()

    return last25testRs

# Plot mean curve and (mean-std, mean+std) curve with some transparency
# Clip the curves to be between 0, 200
def plot_arrays(vars, color, label):
    mean = np.mean(vars, axis=0)
    std = np.std(vars, axis=0)
    plt.plot(range(len(mean)), mean, color=color, label=label)
    plt.fill_between(range(len(mean)), np.maximum(mean-std, 0), np.minimum(mean+std,200), color=color, alpha=0.3)

    # Save the vars array to .npy file
    np.save("reward_cbow.npy", vars)
    
if __name__ == "__main__":
    # Make a flag
    cbow_flag = True

    # Train for different seeds
    curves = []
    for seed in SEEDS:
        curves += [train(seed, cbow_flag)]

    # Plot the curve for the given seeds
    plot_arrays(curves, 'b', 'dqn')
    plt.legend(loc='best')
    plt.show()