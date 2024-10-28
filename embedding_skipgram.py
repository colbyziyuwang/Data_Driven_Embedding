import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SkipGramActionPredictionModel(nn.Module):
    """
    Predict the surrounding actions (previous and next actions) given the current state embedding using a SkipGram-style approach.
    """
    def __init__(self, state_dim, n_actions, embedding_dim=16):
        super(SkipGramActionPredictionModel, self).__init__()

        # state_dim and number of actions
        self.state_dim = state_dim
        self.n_actions = n_actions
        
        # State Embedding for continuous state inputs
        self.state_embedding = nn.Linear(state_dim, embedding_dim)
        
        # Action Embedding for discrete actions
        self.action_embedding = nn.Embedding(n_actions, embedding_dim)

        # Optimizer
        self.optim = torch.optim.Adam(self.parameters(), lr=0.001)
    
    def forward(self, state, context_actions):
        """
        Forward pass: predict the context actions (previous and next) based on the current state embedding.
        
        Args:
            state (torch.Tensor): The current state embedding.
            context_actions (torch.Tensor): The surrounding actions (previous and next actions).
            
        Returns:
            torch.Tensor: The score for context actions.
        """

        # Compute state embedding
        state_embed = self.state_embedding(state)  # Shape: (batch, embed_dim)

        # Compute action embeddings for the context actions
        context_action_embed = self.action_embedding(context_actions)  # Shape: (batch, num_context_actions, embedding_dim)

        # Calculate the score by computing the dot product of the state and context action embeddings
        state_embed = state_embed.unsqueeze(1) # (batch, 1, embed_dim)
        score = torch.bmm(state_embed, context_action_embed.transpose(1, 2))  # Shape: (batch, num_context_actions, 1)

        # Apply log-sigmoid activation to the score
        loss = F.logsigmoid(score).squeeze(-1) # Remove last dimension
        
        return -1 * loss.sum()  # Return the negative log likelihood as the loss

    def train(self, data_loader, epochs=10):
        """
        Train the model on the provided dataset.
        
        Args:
            data_loader (DataLoader): PyTorch DataLoader providing batches of training data.
            epochs (int): Number of training epochs.
        """
        for epoch in range(epochs):
            total_loss = 0
            for batch in data_loader:
                state, context_actions = batch

                # Forward pass: predict the surrounding actions (previous and next)
                loss = self.forward(state, context_actions)
                
                # Backpropagation and optimization
                self.optim.zero_grad()  # Clear the previous gradients
                loss.backward()  # Compute gradients
                self.optim.step()  # Update model parameters
                
                total_loss += loss.item()  # Accumulate the loss for reporting
                
            # print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(data_loader)}")
    
    def compute_distance(self, S):
        """
        Compute the embedding distance between a batch of states S and all actions.
        
        Args:
            S (torch.Tensor): The batch of state feature vectors.
        
        Returns:
            torch.Tensor: The distance matrix between states and all actions, with softmax applied.
        """
        # Get the embeddings for the input states
        states_embeddings = self.state_embedding(S)  # Assume S is of shape (batch_size, state_dim)
        
        # Generate action indices
        action_indices = torch.arange(self.n_actions).to(S.device)
        
        # Get the embeddings for all actions
        action_embeddings = self.action_embedding(action_indices)  # Shape: (n_actions, embedding_size)

        # Compute cosine similarity
        cosine_similarity = F.cosine_similarity(states_embeddings.unsqueeze(1), action_embeddings.unsqueeze(0), dim=-1)
        
        # Apply softmax to get the final distances
        softmax_distance = torch.softmax(cosine_similarity, dim=-1)
        
        return softmax_distance
