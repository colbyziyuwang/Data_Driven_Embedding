import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGramActionPredictionModel(nn.Module):
    """
    Predict the surrounding actions (previous and next actions) given the current state embedding using a SkipGram-style approach.
    """

    def __init__(self, state_dim, n_actions, device, embedding_dim=16):
        super(SkipGramActionPredictionModel, self).__init__()

        self.state_dim = state_dim
        self.n_actions = n_actions
        self.device = device

        # Embedding layers
        self.state_embedding = nn.Linear(state_dim, embedding_dim)
        self.action_embedding = nn.Embedding(n_actions, embedding_dim)

        # Move entire model to device ✅
        self.to(self.device)

        # Optimizer (created after moving model)
        self.optim = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, state, context_actions):
        """
        Forward pass: predict the context actions (previous and next) based on the current state embedding.
        
        Args:
            state (torch.Tensor): Tensor of shape (batch_size, state_dim)
            context_actions (torch.Tensor): Tensor of shape (batch_size, num_context_actions)
        Returns:
            torch.Tensor: Loss (scalar)
        """
        state = state.to(self.device)
        context_actions = context_actions.to(self.device)

        # Compute embeddings
        state_embed = self.state_embedding(state)  # (batch, embed_dim)
        context_embed = self.action_embedding(context_actions)  # (batch, context, embed_dim)

        # Compute score via dot product
        state_embed = state_embed.unsqueeze(1)  # (batch, 1, embed_dim)
        score = torch.bmm(state_embed, context_embed.transpose(1, 2))  # (batch, context, 1)
        loss = F.logsigmoid(score).squeeze(-1)  # (batch, context)

        return -1 * loss.sum()

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
                loss = self.forward(state, context_actions)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                total_loss += loss.item()
            # Optional: print training loss
            # print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(data_loader):.4f}")

    def compute_distance(self, S):
        """
        Compute the cosine similarity between each state and all action embeddings,
        returning a softmax-normalized similarity vector.

        Args:
            S (torch.Tensor): Shape (batch_size, state_dim)
        Returns:
            torch.Tensor: Shape (batch_size, n_actions) — similarity weights
        """
        S = S.to(self.device)
        states_embeddings = self.state_embedding(S)  # (batch, emb_dim)

        action_indices = torch.arange(self.n_actions, device=self.device)
        action_embeddings = self.action_embedding(action_indices)  # (n_actions, emb_dim)

        # Cosine similarity between each state and each action
        cosine_similarity = F.cosine_similarity(
            states_embeddings.unsqueeze(1),        # (batch, 1, emb_dim)
            action_embeddings.unsqueeze(0),        # (1, n_actions, emb_dim)
            dim=-1                                 # (batch, n_actions)
        )

        return F.softmax(cosine_similarity, dim=-1)
