import torch
import pandas as pd
from src.models.mf import MF

ratings = pd.read_parquet("data/processed/features.parquet")
model = MF(n_users=ratings['user_id_enc'].nunique(), n_items=ratings['movie_id_enc'].nunique())
model.load_state_dict(torch.load("saved_model/context_mf.pth"))
model.eval()

# Compute RMSE or Hit@K
