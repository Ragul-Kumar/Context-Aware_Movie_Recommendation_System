import os
import requests
from dotenv import load_dotenv
from serpapi import GoogleSearch
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load API keys
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Endpoints
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"
OMDB_SEARCH_URL = "http://www.omdbapi.com/"
PLACEHOLDER_POSTER = "https://via.placeholder.com/300x450?text=NOPE"


def get_movie_poster_tmdb(title: str) -> str | None:
    if not TMDB_API_KEY:
        return None
    try:
        resp = requests.get(TMDB_SEARCH_URL, params={"api_key": TMDB_API_KEY, "query": title}, timeout=10)
        data = resp.json()
        if data.get("results"):
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return TMDB_IMG_BASE + poster_path
    except Exception as e:
        print(f"TMDb error for {title}: {e}")
    return None


def get_movie_poster_omdb(title: str) -> str | None:
    if not OMDB_API_KEY:
        return None
    try:
        resp = requests.get(OMDB_SEARCH_URL, params={"t": title, "apikey": OMDB_API_KEY}, timeout=10).json()
        poster = resp.get("Poster")
        if poster and poster != "N/A":
            return poster
    except Exception as e:
        print(f"OMDb error for {title}: {e}")
    return None


def get_movie_poster_google(title: str) -> str | None:
    if not SERPAPI_KEY:
        return None
    try:
        search = GoogleSearch({
            "engine": "google",
            "q": f"{title} movie poster",
            "tbm": "isch",
            "ijn": "0",
            "api_key": SERPAPI_KEY
        })
        results = search.get_dict()
        images = results.get("images_results")
        if images:
            return images[0].get("original")
    except Exception as e:
        print(f"Google Images error for {title}: {e}")
    return None


def get_movie_poster(title: str) -> str:
    """
    Parallel poster fetcher: try TMDb, OMDb, Google simultaneously
    Returns first successful poster or placeholder
    """
    funcs = [get_movie_poster_tmdb, get_movie_poster_omdb, get_movie_poster_google]
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_func = {executor.submit(func, title): func for func in funcs}
        for future in as_completed(future_to_func):
            poster = future.result()
            if poster:
                return poster  # return the first successful poster

    return PLACEHOLDER_POSTER
