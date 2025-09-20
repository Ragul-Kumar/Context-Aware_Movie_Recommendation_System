import streamlit as st
import pandas as pd
import torch
import datetime
import pickle
from src.models.mf import MF
from utils import get_movie_poster



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
    model.load_state_dict(torch.load("saved_model/context_mf.pth", map_location="cpu"))
    model.eval()

    with open("models/user_encoder.pkl", "rb") as f:
        user_encoder = pickle.load(f)
    with open("models/movie_encoder.pkl", "rb") as f:
        movie_encoder = pickle.load(f)

    return ratings, movies, model, user_encoder, movie_encoder


ratings, movies, model, user_encoder, movie_encoder = load_model_and_data()


def recommend_top_n(user_id, top_k=10, genre=None):
    user_enc = ratings.loc[ratings['userId'] == user_id, 'user_id_enc'].iloc[0]
    all_items = ratings['movie_id_enc'].unique()

    users = torch.tensor([user_enc] * len(all_items))
    items = torch.tensor(all_items)

    with torch.no_grad():
        scores = model(users, items).numpy()

    score_df = pd.DataFrame({'movie_id_enc': all_items, 'score': scores})
    recommended = ratings[['movie_id_enc', 'movieId']].drop_duplicates()
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
user_id = st.sidebar.number_input(
    "Enter User ID:",
    min_value=1,
    max_value=int(ratings['userId'].max()),
    value=1
)
genre_input = st.sidebar.text_input("Preferred Genre (optional)", "")

# UI style selector
style_choice = st.sidebar.selectbox(
    "🎨 Choose Card Style",
    ["Dark", "Light", "Gradient"]
)

if "history" not in st.session_state:
    st.session_state.history = []


# ===========================
# 🔹 Custom CSS Styles
# ===========================
st.markdown("""
<style>
    .movie-card {
        border-radius: 15px;
        padding: 12px;
        text-align: center;
        transition: transform 0.2s ease-in-out;
        margin-bottom: 12px;
        height: 350px;
        width: 200px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .movie-card:hover {
        transform: scale(1.05);
    }
    .dark-card {
        background: linear-gradient(145deg, #1E1E2F, #2C2F33);
        box-shadow: 0 4px 10px rgba(0,0,0,0.4);
        color: white;
    }
    .light-card {
        background: #f9f9f9;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        color: black;
    }
    .gradient-card {
        background: linear-gradient(135deg, #ff4b4b, #ff944b);
        color: white;
        box-shadow: 0 4px 15px rgba(255,75,75,0.4);
    }
    .movie-title {
        font-size: 1rem;
        font-weight: bold;
        margin-top: 10px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .movie-genre {
        font-size: 0.85rem;
        opacity: 0.8;
    }
</style>
""", unsafe_allow_html=True)

# Map choice → CSS class
card_class = {
    "Dark": "dark-card",
    "Light": "light-card",
    "Gradient": "gradient-card"
}[style_choice]


# ===========================
# 🔹 Get recommendations
# ===========================
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
            "results": recs[['title', 'genres']].to_dict(orient="records")
        })

        st.subheader(f"🎬 Top 10 Recommendations for User {user_id} ({genre_input if genre_input else 'All Genres'})")

        # Display recommendations in horizontal grid (4 columns per row)
        cards_per_row = 4
        poster_height = "225px"

        for row_idx in range(0, len(recs), cards_per_row):
            cols = st.columns(cards_per_row)
            for col_idx, row in enumerate(recs.iloc[row_idx:row_idx + cards_per_row].itertuples()):
                poster_url = get_movie_poster(row.title)
                with cols[col_idx]:
                    if poster_url and "placeholder" not in poster_url:
                        poster_html = f'<img src="{poster_url}" style="width:100%; height:{poster_height}; object-fit:cover; border-radius:8px;"/>'
                    else:
                        poster_html = f'<div style="width:100%; height:{poster_height}; display:flex; align-items:center; justify-content:center; background-color:#555; color:white; border-radius:8px;">NO POSTER</div>'

                    st.markdown(
                        f"""
                        <div class="movie-card {card_class}">
                            <div style="width:100%; flex-shrink:0; margin-bottom:10px;">
                                {poster_html}
                            </div>
                            <div style="width:100%; text-align:center;">
                                <div class="movie-title">{row.title}</div>
                                <div class="movie-genre">{row.genres.replace('|', ', ')}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# ===========================
# 🔹 Search history
# ===========================
if st.session_state.history:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🕒 Search History (Last 5)")
    for entry in reversed(st.session_state.history[-5:]):  # show last 5
        if st.sidebar.button(f"{entry['timestamp']} | User {entry['user_id']} ({entry['genre']})"):
            st.subheader(f"📜 Past Results for User {entry['user_id']} ({entry['genre']})")
            for i, res in enumerate(entry['results'], 1):
                st.markdown(f"**{i}. {res['title']}** — _{res['genres']}_")
