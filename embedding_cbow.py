import torch
import torch.nn as nn
import torch.nn.functional as F

class StateActionPredictionModel(nn.Module):
    """
    A model for predicting actions given the context of previous states and actions using techniques similar to CBOW.
    
    Args:
        state_dim (int): Dimensionality of the state input (continuous).
        n_actions (int): Number of possible discrete actions.
        embedding_size (int): Dimensionality of the embeddings for both states and actions.
        context_size (int): Size of the context, i.e., how many state, action elements.
        device (torch.device): The device to run the model on (CPU or CUDA).
    """
    
    def __init__(self, state_dim, n_actions, device, embedding_size=16, context_size=5):
        super(StateActionPredictionModel, self).__init__()
        
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.device = device

        # Embedding layers
        self.state_embedding = nn.Linear(state_dim, embedding_size)
        self.action_embedding = nn.Embedding(n_actions, embedding_size)

        # Fully connected layers to predict the missing action
        self.fc = nn.Sequential(
            nn.Linear(embedding_size * context_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

        # ✅ Move the entire model to the desired device
        self.to(self.device)

        # Loss function and optimizer (after model is on device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, s1, a1, s2, s3, a3):
        """
        Forward pass for predicting the missing action from context.
        """
        # Embed states and actions
        s1_embed = self.state_embedding(s1.to(self.device))
        a1_embed = self.action_embedding(a1.to(self.device))
        s2_embed = self.state_embedding(s2.to(self.device))
        s3_embed = self.state_embedding(s3.to(self.device))
        a3_embed = self.action_embedding(a3.to(self.device))

        # Concatenate the context
        context = torch.cat([s1_embed, a1_embed, s2_embed, s3_embed, a3_embed], dim=-1)

        # Predict the action
        return self.fc(context)

    def train(self, data_loader, epochs=10):
        """
        Train the model using CBOW-style prediction of action from surrounding context.
        """
        for epoch in range(epochs):
            total_loss = 0
            for batch in data_loader:
                s1, a1, s2, s3, a3, target_action = [x.to(self.device) for x in batch]

                # Forward pass
                pred_action = self.forward(s1, a1, s2, s3, a3)

                # Compute loss
                loss = self.loss_fn(pred_action, target_action)

                # Backpropagation
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                total_loss += loss.item()
            # Uncomment to monitor training loss
            # print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(data_loader):.4f}")

    def compute_distance(self, S):
        """
        Compute softmax-normalized cosine similarity between state embeddings and all action embeddings.
        Returns a weight vector over actions for each input state.
        """
        S = S.to(self.device)
        states_embeddings = self.state_embedding(S)  # (batch_size, emb_dim)

        # Get embeddings for all actions
        action_indices = torch.arange(self.n_actions, device=self.device)
        action_embeddings = self.action_embedding(action_indices)  # (n_actions, emb_dim)

        # Compute cosine similarity
        cosine_similarity = F.cosine_similarity(
            states_embeddings.unsqueeze(1),         # (batch_size, 1, emb_dim)
            action_embeddings.unsqueeze(0),         # (1, n_actions, emb_dim)
            dim=-1                                  # → (batch_size, n_actions)
        )

        # Softmax over actions
        return F.softmax(cosine_similarity, dim=-1)
