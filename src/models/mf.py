import torch
import torch.nn as nn

class MF(nn.Module):
    def __init__(self, num_users, num_items, emb_size=64):
        super(MF, self).__init__()
        # Embedding layers for users and items
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)

        # Initialize embeddings with small random values
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user, item):
        """
        user: tensor of user indices
        item: tensor of item indices
        returns: predicted rating (dot product of embeddings)
        """
        user_vec = self.user_emb(user)
        item_vec = self.item_emb(item)

        # Dot product between user and item vectors
        pred = (user_vec * item_vec).sum(dim=1)
        return pred
