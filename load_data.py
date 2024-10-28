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

class SkipGramStateActionDataset(Dataset):
    def __init__(self, state_action_sequences):
        """
        Dataset class to handle sequences of state-action pairs.
        Each sequence is of the form (a1, s1, a2).
        
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
        The item is a tuple of (a1, s2, a2), where a1, a2 are the actions to predict.

        Args:
            idx: Index of the sample to return.
        
        Returns:
            Tuple of tensors (a1, s2, a2)
        """
        a1 = torch.tensor(self.sequences[idx][0], dtype=torch.long)    # Action a1 (target action to predict)
        s2 = torch.tensor(check_s(self.sequences[idx][1]), dtype=torch.float32)  # State s1
        a2 = torch.tensor(self.sequences[idx][2], dtype=torch.long)    # Action a2 (target action to predict)
        
        # Now return both actions as a stacked tensor (shape 2, embed_dim)
        action_tensor = torch.stack([a1, a2])  # Stack a1 and a2
        
        return s2, action_tensor # Return the state and context actions