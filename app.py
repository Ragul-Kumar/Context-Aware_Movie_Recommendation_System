# app.py
import streamlit as st
import pandas as pd
import torch
import datetime
import pickle
from src.models.mf import MF


@st.cache_resource
def load_model_and_data():
    ratings = pd.read_parquet("data/processed/features.parquet")

    movies = pd.read_csv(
        "data/raw/ml-1m/movies.dat",
        sep="::", engine="python",
        header=None, names=["movieId", "title", "genres"],
        encoding="latin-1"
    )

    n_users = ratings['user_id_enc'].nunique()
    n_items = ratings['movie_id_enc'].nunique()
    model = MF(num_users=n_users, num_items=n_items, emb_size=64)
    model.load_state_dict(torch.load("saved_model/context_mf.pth"))
    model.eval()

    with open("models/user_encoder.pkl", "rb") as f:
        user_encoder = pickle.load(f)
    with open("models/movie_encoder.pkl", "rb") as f:
        movie_encoder = pickle.load(f)

    return ratings, movies, model, user_encoder, movie_encoder


ratings, movies, model, user_encoder, movie_encoder = load_model_and_data()

def recommend_top_n(user_id, top_k=10, genre=None):
    user_enc = ratings.loc[ratings['userId']==user_id, 'user_id_enc'].iloc[0]
    all_items = ratings['movie_id_enc'].unique()

    users = torch.tensor([user_enc]*len(all_items))
    items = torch.tensor(all_items)

    with torch.no_grad():
        scores = model(users, items).numpy()

    score_df = pd.DataFrame({'movie_id_enc': all_items, 'score': scores})
    recommended = ratings[['movie_id_enc','movieId']].drop_duplicates()
    recommended = recommended.merge(movies, on='movieId')
    recommended = recommended.merge(score_df, on='movie_id_enc')

    if genre:
        recommended = recommended[recommended['genres'].str.contains(genre, case=False, na=False)]

    recommended = recommended.sort_values('score', ascending=False).head(top_k)
    return recommended


st.set_page_config(page_title="Movie Recommender 🎬", layout="wide")

st.title("🍿 Movie Recommendation System")
st.write("Personalized movie suggestions based on your past interactions.")

# Sidebar
st.sidebar.header("🔍 Search Settings")
user_id = st.sidebar.number_input("Enter User ID:", min_value=1, max_value=int(ratings['userId'].max()), value=1)
genre_input = st.sidebar.text_input("Preferred Genre (optional)", "")


if "history" not in st.session_state:
    st.session_state.history = []

if st.sidebar.button("✨ Get Recommendations"):
    recs = recommend_top_n(user_id=user_id, top_k=10, genre=genre_input if genre_input else None)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if recs.empty:
        st.warning("⚠️ No recommendations found. Try another genre or user ID.")
    else:
        # Save search history
        st.session_state.history.append({
            "timestamp": now,
            "user_id": user_id,
            "genre": genre_input if genre_input else "Any",
            "results": recs[['title','genres']].to_dict(orient="records")
        })

        st.subheader(f"🎬 Top 10 Recommendations for User {user_id} ({genre_input if genre_input else 'All Genres'})")
        
        # Display as nice cards
        for i, row in enumerate(recs.itertuples(), 1):
            with st.container():
                st.markdown(
                    f"""
                    <div style="padding:10px; margin-bottom:10px; border-radius:12px; 
                                background-color:#f8f9fa; box-shadow:0 2px 6px rgba(0,0,0,0.1);">
                        <h4 style="margin:0; color:#1E90FF;">{i}. {row.title}</h4>
                        <p style="margin:0; color:#6c757d;">Genres: {row.genres.replace('|', ', ')}</p>
                    </div>
                    """, unsafe_allow_html=True
                )

if st.session_state.history:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🕒 Search History")
    for entry in reversed(st.session_state.history[-5:]):  # show last 5
        st.sidebar.markdown(
            f"""
            **{entry['timestamp']}**  
            👤 User: {entry['user_id']}  
            🎭 Genre: {entry['genre']}  
            🎬 Top: {entry['results'][0]['title']}  
            """, unsafe_allow_html=True
        )
