import torch
import torch.nn as nn

class ContextAwareMF(nn.Module):
    """
    Slightly different flavor: elementwise product of user/item + context MLP
    forward(user, item, context) where context shape is (batch, 2) [hour_norm, day_norm]
    """
    def __init__(self, n_users, n_items, n_factors=32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)
        self.fc_context = nn.Linear(2, n_factors)  # hour + day -> factor
        self.out = nn.Linear(n_factors, 1)

    def forward(self, user, item, context):
        # user/item embeddings
        u = self.user_emb(user)
        v = self.item_emb(item)
        x = u * v                    # interaction (batch, n_factors)
        c = self.fc_context(context) # (batch, n_factors)
        x = x + c
        return self.out(x).squeeze(-1)
