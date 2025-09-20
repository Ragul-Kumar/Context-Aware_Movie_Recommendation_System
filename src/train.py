# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import os

from src.models.mf import MF


class RatingsDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['user_id_enc'].values, dtype=torch.long)
        self.movies = torch.tensor(df['movie_id_enc'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]


def train_context_mf(features_parquet, saved_model_dir="saved_model", epochs=5, batch_size=512, lr=1e-3):
    # Load features
    df = pd.read_parquet(features_parquet)

    dataset = RatingsDataset(df)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_users = df['user_id_enc'].nunique()
    num_movies = df['movie_id_enc'].nunique()

    model = MF(num_users, num_movies, emb_size=64)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    os.makedirs(saved_model_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        total_loss = 0
        for users, movies, ratings in loader:
            optimizer.zero_grad()
            preds = model(users, movies).squeeze()
            loss = criterion(preds, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs}: Avg Loss={avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), os.path.join(saved_model_dir, "context_mf.pth"))
    return model
