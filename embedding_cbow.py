import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class StateActionPredictionModel(nn.Module):
    """
    A model for predicting actions given the context of previous states and actions using techniques similar to CBOW.
    
    Args:
        state_dim (int): Dimensionality of the state input (continuous).
        n_actions (int): Number of possible discrete actions.
        embedding_size (int): Dimensionality of the embeddings for both states and actions.
        context_size (int): Size of the context, i.e., how many state-action pairs are used as input for predicting the missing action.
    """
    
    def __init__(self, state_dim, n_actions, embedding_size=16, context_size=5):
        super(StateActionPredictionModel, self).__init__()
        
        # State Dimension and Number of Actions
        self.state_dim = state_dim
        self.n_actions = n_actions

        # Embedding layers for states (continuous) and actions (discrete)
        self.state_embedding = nn.Linear(state_dim, embedding_size)  # Linear layer for continuous state embeddings
        self.action_embedding = nn.Embedding(n_actions, embedding_size)  # Embedding layer for discrete action embeddings
        
        # Fully connected layers to predict the missing action from the context
        self.fc = nn.Sequential(
            nn.Linear(embedding_size * context_size, 128),  # Input size: context embedding size
            nn.ReLU(),  # Non-linear activation function
            nn.Linear(128, n_actions)  # Output size: number of actions
        )

        # Loss function for multi-class classification
        self.loss_fn = nn.CrossEntropyLoss()

        # Optimizer for training
        self.optim = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, s1, a1, s2, s3, a3):
        """
        Forward pass for the model.
        
        Args:
            s1 (torch.Tensor): State embedding for the first state in the context.
            a1 (torch.Tensor): Action embedding for the first action in the context.
            s2 (torch.Tensor): State embedding for the second state in the context.
            s3 (torch.Tensor): State embedding for the third state in the context.
            a3 (torch.Tensor): Action embedding for the third action in the context.
        
        Returns:
            torch.Tensor: Predicted action logits.
        """
        # Embed states and actions
        s1_embed = self.state_embedding(s1)
        a1_embed = self.action_embedding(a1)
        s2_embed = self.state_embedding(s2)
        s3_embed = self.state_embedding(s3)
        a3_embed = self.action_embedding(a3)
        
        # Concatenate the embeddings from the context
        context = torch.cat([s1_embed, a1_embed, s2_embed, s3_embed, a3_embed], dim=-1)
        
        # Pass the context embeddings through the fully connected layers to predict the missing action
        action_prediction = self.fc(context)
        
        return action_prediction

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
                s1, a1, s2, s3, a3, target_action = batch
                
                # Forward pass: predict the missing action (a2)
                pred_action = self.forward(s1, a1, s2, s3, a3)
                
                # Compute the loss between the predicted and target action
                loss = self.loss_fn(pred_action, target_action)
                
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
        action_embeddings = self.action_embedding(action_indices)  # Shape (n_actions, embedding_size)

        # Compute cosine similarity
        cosine_similarity = F.cosine_similarity(states_embeddings.unsqueeze(1), action_embeddings.unsqueeze(0), dim=-1)
        
        # Apply softmax to get the final distances
        softmax_distance = torch.softmax(cosine_similarity, dim=-1)
        
        return softmax_distance
