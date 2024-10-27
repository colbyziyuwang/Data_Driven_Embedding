import torch
from torch.utils.data import Dataset, DataLoader

# Check if s is a tuple
def check_s(s):
    if (isinstance(s, tuple)):
        s = s[0]
    return s
class StateActionDataset(Dataset):
    def __init__(self, state_action_sequences):
        """
        Dataset class to handle sequences of state-action pairs.
        Each sequence is of the form (s1, a1, s2, a2, s3, a3).
        
        Args:
            state_action_sequences: List of sequences, where each sequence is a tuple/list of (s1, a1, s2, a2, s3, a3).
        """
        self.sequences = state_action_sequences  # A list of sequences, each in the format [s1, a1, s2, a2, s3, a3]

    def __len__(self):
        """
        Return the length of the dataset.
        """
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Return a single item from the dataset at index `idx`.
        The item is a tuple of (s1, a1, s2, s3, a3, a2), where a2 is the action to predict.

        Args:
            idx: Index of the sample to return.
        
        Returns:
            Tuple of tensors (s1, a1, s2, s3, a3, a2)
        """
        s1 = torch.tensor(check_s(self.sequences[idx][0]), dtype=torch.float32)  # State s1
        a1 = torch.tensor(self.sequences[idx][1], dtype=torch.long)    # Action a1
        s2 = torch.tensor(check_s(self.sequences[idx][2]), dtype=torch.float32)  # State s2
        a2 = torch.tensor(self.sequences[idx][3], dtype=torch.long)    # Action a2 (target action to predict)
        s3 = torch.tensor(check_s(self.sequences[idx][4]), dtype=torch.float32)  # State s3
        a3 = torch.tensor(self.sequences[idx][5], dtype=torch.long)    # Action a3
        
        return s1, a1, s2, s3, a3, a2  # Return the context and target action


# Example Usage

# # Example state-action sequences (you should replace this with your actual dataset)
# # Each sequence is of the form [s1, a1, s2, a2, s3, a3]
# state_action_sequences = [
#     # Example data
#     ([0.1, 0.2, 0.3], 1, [0.4, 0.5, 0.6], 2, [0.7, 0.8, 0.9], 3),
#     ([0.2, 0.3, 0.4], 2, [0.5, 0.6, 0.7], 1, [0.8, 0.9, 1.0], 0),
#     # Add more sequences as needed
# ]

# # Initialize dataset
# dataset = StateActionDataset(state_action_sequences)

# # Initialize DataLoader
# data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# # Example usage: Iterate through the DataLoader
# for batch in data_loader:
#     s1, a1, s2, s3, a3, target_action = batch
#     print(f"s1: {s1}, a1: {a1}, s2: {s2}, s3: {s3}, a3: {a3}, target: {target_action}")
