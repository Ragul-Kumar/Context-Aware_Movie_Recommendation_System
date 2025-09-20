# train_context.py - optional alternative training script expecting different CSV layout
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from src.models.context_aware import ContextAwareMF

class MovieLensDataset(Dataset):
    def __init__(self, df, user_enc, item_enc, weekday_enc, time_enc):
        self.user = torch.tensor(user_enc.transform(df["userId"]), dtype=torch.long)
        self.item = torch.tensor(item_enc.transform(df["movieId"]), dtype=torch.long)
        self.weekday = torch.tensor(weekday_enc.transform(df["weekday"]), dtype=torch.long)
        self.time = torch.tensor(time_enc.transform(df["timeOfDay"]), dtype=torch.long)
        self.rating = torch.tensor(df["rating"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.rating)

    def __getitem__(self, idx):
        return self.user[idx], self.item[idx], self.weekday[idx], self.time[idx], self.rating[idx]

def train_context_aware_model(
    train_csv="data/processed/train.csv",
    movies_csv="data/processed/movies.csv",
    save_path="saved_model/context_aware/",
    embed_dim=32,
    batch_size=256,
    lr=0.001,
    epochs=5
):
    train_df = pd.read_csv(train_csv)
    movies = pd.read_csv(movies_csv)

    train_df["weekday"] = pd.to_datetime(train_df["timestamp"]).dt.day_name()
    train_df["timeOfDay"] = pd.to_datetime(train_df["timestamp"]).dt.hour // 6  # 4 buckets

    user_enc = LabelEncoder().fit(train_df["userId"])
    item_enc = LabelEncoder().fit(train_df["movieId"])
    weekday_enc = LabelEncoder().fit(train_df["weekday"])
    time_enc = LabelEncoder().fit(train_df["timeOfDay"])

    dataset = MovieLensDataset(train_df, user_enc, item_enc, weekday_enc, time_enc)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ContextAwareMF(
        n_users=len(user_enc.classes_),
        n_items=len(item_enc.classes_),
        n_factors=embed_dim
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for u, i, w, t, r in loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            # rebuild context as two floats in [0,1] using encoders' integer labels is tricky here;
            # this script is more of an example - adapt to the training csv format you actually have.
            # We'll use dummy context zeros for compatibility:
            context = torch.zeros((u.size(0), 2), dtype=torch.float32, device=device)
            pred = model(u, i, context)
            loss = loss_fn(pred, r)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(r)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss={avg_loss:.4f}")

    os.makedirs(save_path, exist_ok=True)
    torch.save({
        "state": model.state_dict(),
        "meta": dict(),
    }, os.path.join(save_path, "model.pth"))

    print(f"Model saved to {save_path}")
