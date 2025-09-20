import torch
import torch.nn as nn

class GRU4Rec(nn.Module):
    def __init__(self, n_items, embed_size=32, hidden_size=64):
        super(GRU4Rec, self).__init__()
        self.item_emb = nn.Embedding(n_items, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, n_items)

    def forward(self, x):
        x = self.item_emb(x)    # (batch, seq_len, embed)
        _, h = self.gru(x)
        return self.out(h.squeeze(0))
