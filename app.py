import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch — Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# TMDB CONFIG  →  paste your free API key here
# ─────────────────────────────────────────────
TMDB_API_KEY = "a16fd305bad486e7fde40d4afdc99c69"   # <── replace this
TMDB_BASE    = "https://api.themoviedb.org/3"
POSTER_BASE  = "https://image.tmdb.org/t/p/w500"
BACKDROP_BASE= "https://image.tmdb.org/t/p/w1280"

# ─────────────────────────────────────────────
# CUSTOM CSS  — cinematic dark theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    color: #e8e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: #0f0f1a !important;
    border-right: 1px solid #1e1e2e !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Top nav bar ── */
.topbar {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 0 28px 0;
    border-bottom: 1px solid #1e1e2e;
    margin-bottom: 32px;
}
.topbar-logo {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.2rem;
    letter-spacing: 3px;
    background: linear-gradient(135deg, #e50914 0%, #ff6b35 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.topbar-tagline {
    font-size: 0.78rem;
    color: #666;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── Section headers ── */
.section-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.6rem;
    letter-spacing: 2px;
    color: #e8e8f0;
    margin: 32px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #e50914 0%, transparent 100%);
    margin-left: 12px;
}

/* ── Movie card ── */
.movie-card {
    background: #111120;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #1e1e2e;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    cursor: pointer;
    height: 100%;
}
.movie-card:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 20px 40px rgba(229,9,20,0.25);
    border-color: #e50914;
}
.movie-card img {
    width: 100%;
    aspect-ratio: 2/3;
    object-fit: cover;
    display: block;
}
.movie-card-body {
    padding: 12px 14px 14px;
}
.movie-card-title {
    font-weight: 600;
    font-size: 0.9rem;
    color: #e8e8f0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 6px;
}
.movie-card-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.78rem;
    color: #888;
}
.rating-badge {
    background: linear-gradient(135deg, #e50914, #ff6b35);
    color: white;
    font-weight: 700;
    font-size: 0.72rem;
    padding: 2px 7px;
    border-radius: 4px;
}
.genre-pill {
    background: #1e1e2e;
    color: #aaa;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 20px;
    border: 1px solid #2e2e3e;
    white-space: nowrap;
}

/* ── Hero banner ── */
.hero-banner {
    border-radius: 16px;
    overflow: hidden;
    position: relative;
    margin-bottom: 32px;
    min-height: 340px;
}
.hero-banner img {
    width: 100%;
    height: 340px;
    object-fit: cover;
    display: block;
    filter: brightness(0.45);
}
.hero-overlay {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    padding: 32px 36px;
    background: linear-gradient(0deg, rgba(10,10,15,0.95) 0%, transparent 100%);
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3rem;
    letter-spacing: 3px;
    line-height: 1;
    color: #fff;
    margin-bottom: 10px;
}
.hero-desc {
    font-size: 0.9rem;
    color: #bbb;
    max-width: 600px;
    line-height: 1.6;
    margin-bottom: 14px;
}
.hero-badges { display: flex; gap: 8px; flex-wrap: wrap; }

/* ── Reason chip ── */
.reason-chip {
    background: #1a1a2e;
    border: 1px solid #e50914;
    color: #ff6b35;
    font-size: 0.72rem;
    padding: 4px 12px;
    border-radius: 20px;
    display: inline-block;
    margin-bottom: 12px;
}

/* ── Watchlist badge ── */
.watchlist-count {
    background: #e50914;
    color: white;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 20px;
    vertical-align: middle;
    margin-left: 6px;
}

/* ── Streamlit widget overrides ── */
[data-testid="stSelectbox"] label,
[data-testid="stTextInput"] label,
[data-testid="stSlider"] label,
[data-testid="stMultiSelect"] label {
    color: #aaa !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.5px !important;
}
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] > div > div > input {
    background: #111120 !important;
    border: 1px solid #2e2e3e !important;
    color: #e8e8f0 !important;
    border-radius: 8px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #e50914, #c0000e) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    padding: 8px 20px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

div[data-testid="stHorizontalBlock"] { gap: 16px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TMDB HELPERS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def tmdb_get(endpoint, params={}):
    params["api_key"] = TMDB_API_KEY
    try:
        r = requests.get(f"{TMDB_BASE}/{endpoint}", params=params, timeout=6)
        return r.json() if r.ok else {}
    except Exception:
        return {}

def poster_url(path):
    return f"{POSTER_BASE}{path}" if path else "https://via.placeholder.com/300x450/111120/444?text=No+Poster"

def backdrop_url(path):
    return f"{BACKDROP_BASE}{path}" if path else None

@st.cache_data(show_spinner=False)
def search_tmdb(query):
    data = tmdb_get("search/movie", {"query": query, "language": "en-US", "page": 1})
    return data.get("results", [])[:8]

@st.cache_data(show_spinner=False)
def get_trending():
    data = tmdb_get("trending/movie/week")
    return data.get("results", [])[:10]

@st.cache_data(show_spinner=False)
def get_top_rated():
    data = tmdb_get("movie/top_rated", {"language": "en-US", "page": 1})
    return data.get("results", [])[:10]

@st.cache_data(show_spinner=False)
def get_movie_details(movie_id):
    return tmdb_get(f"movie/{movie_id}", {"language": "en-US", "append_to_response": "videos,credits"})

@st.cache_data(show_spinner=False)
def get_watch_providers(movie_id):
    data = tmdb_get(f"movie/{movie_id}/watch/providers")
    return data.get("results", {}).get("IN", {})   # change "IN" to your country code


# ─────────────────────────────────────────────
# RECOMMENDATION ENGINE  (hybrid TF-IDF + weighted score)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_sample_data():
    # ── Load both CSV files ──────────────────────────────────────────
    movies  = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    # ── The credits file uses "movie_id" OR "id" depending on version
    #    Rename so both files share the same key column "id"
    if "movie_id" in credits.columns:
        credits.rename(columns={"movie_id": "id"}, inplace=True)

    # ── Merge on title (safest — works for both versions of credits CSV)
    movies = movies.merge(credits, on="title", how="left")

    # ── If there are duplicate id columns after merge, drop the extra one
    if "id_x" in movies.columns:
        movies.rename(columns={"id_x": "id"}, inplace=True)
        movies.drop(columns=["id_y"], inplace=True, errors="ignore")

    # ── Keep only the columns we need ───────────────────────────────
    movies = movies[[
        "id", "title", "overview", "genres", "keywords",
        "cast", "crew", "vote_average", "popularity", "release_date"
    ]].copy()

    movies.dropna(subset=["title", "overview"], inplace=True)

    # ── Helper: parse JSON columns like genres, keywords ────────────
    def parse_names(text):
        try:
            items = ast.literal_eval(str(text))
            return " ".join([x["name"] for x in items])
        except Exception:
            return ""

    # ── Helper: get top 3 cast names (no spaces so TF-IDF treats as 1 token)
    def parse_cast(text):
        try:
            items = ast.literal_eval(str(text))
            return " ".join([x["name"].replace(" ", "") for x in items[:3]])
        except Exception:
            return ""

    # ── Helper: get director name from crew list ─────────────────────
    def parse_director(text):
        try:
            for x in ast.literal_eval(str(text)):
                if x.get("job") == "Director":
                    return x["name"].replace(" ", "")
            return ""
        except Exception:
            return ""

    movies["genres"]   = movies["genres"].apply(parse_names)
    movies["keywords"] = movies["keywords"].apply(parse_names)
    movies["cast"]     = movies["cast"].apply(parse_cast)
    movies["director"] = movies["crew"].apply(parse_director)

    # ── Build the "soup" — everything TF-IDF will learn from ─────────
    movies["soup"] = (
        movies["overview"].fillna("")  + " " +
        movies["genres"].fillna("")    + " " +
        movies["keywords"].fillna("")  + " " +
        movies["cast"].fillna("")      + " " +
        movies["director"].fillna("")
    )

    # ── Keep release year only (cleaner display) ─────────────────────
    movies["release_date"] = movies["release_date"].astype(str).str[:4]

    movies.reset_index(drop=True, inplace=True)
    return movies

@st.cache_data(show_spinner=False)
def build_recommender(df):
    df = df.copy()
    # Use the full rich soup built in load_sample_data (overview + genres + keywords + cast + director)
    # Weight genres and keywords more heavily by repeating them — so a sci-fi film
    # strongly matches other sci-fi films even if plot words differ
    df["weighted_soup"] = (
        df["overview"].fillna("")   + " " +
        df["genres"].fillna("")     + " " +
        df["genres"].fillna("")     + " " +   # genres twice = 2x weight
        df["keywords"].fillna("")   + " " +
        df["keywords"].fillna("")   + " " +   # keywords twice = 2x weight
        df["cast"].fillna("")       + " " +
        df["director"].fillna("")   + " " +
        df["director"].fillna("")       # director twice — same director = strong signal
    )
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=15000)
    tfidf_matrix = tfidf.fit_transform(df["weighted_soup"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def get_recommendations(title, df, cosine_sim, top_n=8):
    if title not in df["title"].values:
        return pd.DataFrame(), ""

    idx = df[df["title"] == title].index[0]
    content_scores = list(enumerate(cosine_sim[idx]))

    # ── Log-scale popularity so outliers (e.g. Interstellar=724, mean=21)
    #    don't drown out the content similarity signal ──────────────────
    import numpy as np
    log_pop  = np.log1p(df["popularity"])
    log_vote = np.log1p(df["vote_average"])
    max_log_pop  = log_pop.max()
    max_log_vote = log_vote.max()

    # ── Step 1: compute hybrid score for every movie ──────────────────
    scored = []
    for i, cs in content_scores:
        if i == idx:
            continue
        pop_score    = log_pop.iloc[i]  / max_log_pop
        rating_score = log_vote.iloc[i] / max_log_vote
        # Content similarity is now the dominant signal (65%)
        hybrid_score = 0.65 * cs + 0.20 * rating_score + 0.15 * pop_score
        scored.append((i, hybrid_score, cs))

    scored = sorted(scored, key=lambda x: x[1], reverse=True)

    # ── Step 2: diversity filter — max 2 movies per director ──────────
    src_director = df.iloc[idx]["director"]
    director_count = {}
    diverse = []
    for i, score, cs in scored:
        director = df.iloc[i]["director"]
        # Limit same director to 2 results (unless it IS the source director's other films)
        if director and director != src_director:
            if director_count.get(director, 0) >= 2:
                continue
            director_count[director] = director_count.get(director, 0) + 1
        diverse.append((i, score))
        if len(diverse) == top_n:
            break

    # ── Step 3: build result dataframe ───────────────────────────────
    recs = df.iloc[[i for i, _ in diverse]].copy()
    recs["match_score"] = [round(s * 100) for _, s in diverse]

    # ── Step 4: explainability — why was this recommended? ────────────
    src_genres   = set(df.iloc[idx]["genres"].split())
    src_cast     = set(df.iloc[idx]["cast"].split())
    src_director = df.iloc[idx]["director"]
    reasons = []
    for i, _ in diverse:
        rec = df.iloc[i]
        rec_genres = set(rec["genres"].split())
        rec_cast   = set(rec["cast"].split())
        shared_genres = src_genres & rec_genres
        shared_cast   = src_cast   & rec_cast
        if rec["director"] == src_director and src_director:
            reasons.append(f"Same director: {src_director.replace('', ' ').strip()}")
        elif shared_cast:
            actor = list(shared_cast)[0]
            # restore spaces in the concatenated name (e.g. LeonardoDiCaprio → Leonardo DiCaprio)
            import re
            actor_readable = re.sub(r'([a-z])([A-Z])', r'\1 \2', actor)
            reasons.append(f"Stars {actor_readable}")
        elif shared_genres:
            reasons.append(", ".join(list(shared_genres)[:2]))
        else:
            reasons.append("Similar themes")
    recs["reason"] = reasons
    top_genre = list(src_genres)[0] if src_genres else ""
    return recs, top_genre


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None
if "search_query" not in st.session_state:
    st.session_state.search_query = ""


# ─────────────────────────────────────────────
# TOP NAVBAR  (no sidebar needed)
# ─────────────────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:center; justify-content:space-between;
            padding: 16px 0 20px 0; border-bottom: 1px solid #1e1e2e; margin-bottom: 28px;">
    <div>
        <span style="font-family:'Bebas Neue',sans-serif; font-size:2rem; letter-spacing:3px;
                     background:linear-gradient(135deg,#e50914,#ff6b35);
                     -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            CineMatch
        </span>
        <span style="font-size:0.72rem; color:#555; letter-spacing:2px;
                     text-transform:uppercase; margin-left:12px;">
            Movie Recommender
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation tabs — always visible at the top
tab1, tab2, tab3 = st.tabs(["🏠  Home", "🔍  Search & Recommend", "📋  My Watchlist"])

st.markdown("""
<style>
button[data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: #888 !important;
    padding: 8px 24px !important;
    background: transparent !important;
    border: none !important;
    letter-spacing: 0.3px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #e8e8f0 !important;
    border-bottom: 2px solid #e50914 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────────
df = load_sample_data()
cosine_sim = build_recommender(df)


# ─────────────────────────────────────────────
# HELPER: render movie grid from TMDB results
# ─────────────────────────────────────────────
def render_tmdb_grid(movies_list, cols=5):
    columns = st.columns(cols)
    for i, movie in enumerate(movies_list[:cols*2]):
        with columns[i % cols]:
            p = poster_url(movie.get("poster_path"))
            rating = movie.get("vote_average", 0)
            year   = str(movie.get("release_date", ""))[:4]
            title  = movie.get("title", "Unknown")
            genres_raw = movie.get("genre_ids", [])

            st.markdown(f"""
            <div class="movie-card">
                <img src="{p}" alt="{title}" loading="lazy"/>
                <div class="movie-card-body">
                    <div class="movie-card-title">{title}</div>
                    <div class="movie-card-meta">
                        <span class="rating-badge">★ {rating:.1f}</span>
                        <span>{year}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_local_grid(recs_df, cols=4):
    columns = st.columns(cols)
    for i, (_, row) in enumerate(recs_df.iterrows()):
        with columns[i % cols]:
            results = search_tmdb(row["title"])
            p = poster_url(results[0].get("poster_path") if results else None)
            mid = results[0].get("id") if results else None

            reason_html = f'<div class="reason-chip">✦ {row["reason"]}</div>' if "reason" in row else ""
            score_html  = f'<div class="rating-badge">Match {row["match_score"]}%</div>' if "match_score" in row else ""

            st.markdown(f"""
            <div class="movie-card">
                <img src="{p}" alt="{row['title']}" loading="lazy"/>
                <div class="movie-card-body">
                    {reason_html}
                    <div class="movie-card-title">{row['title']}</div>
                    <div class="movie-card-meta">
                        <span class="rating-badge">★ {row['vote_average']}</span>
                        <span>{row['release_date']}</span>
                        {score_html}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("＋ Watch", key=f"wl_{i}_{row['title']}"):
                    if row["title"] not in st.session_state.watchlist:
                        st.session_state.watchlist.append(row["title"])
                        st.toast(f"Added {row['title']} to watchlist!", icon="✅")
            with col_b:
                if mid and st.button("Details", key=f"dt_{i}_{row['title']}"):
                    st.session_state.selected_movie = mid


# ─────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────
with tab1:
    # Hero — first trending movie
    with st.spinner("Loading trending movies..."):
        trending = get_trending()

    if trending:
        hero = trending[0]
        bd = backdrop_url(hero.get("backdrop_path"))
        if bd:
            st.markdown(f"""
            <div class="hero-banner">
                <img src="{bd}" alt="hero"/>
                <div class="hero-overlay">
                    <div class="hero-title">{hero.get('title','')}</div>
                    <div class="hero-desc">{hero.get('overview','')[:220]}...</div>
                    <div class="hero-badges">
                        <span class="rating-badge">★ {hero.get('vote_average',0):.1f}</span>
                        <span class="genre-pill">🔥 Trending</span>
                        <span class="genre-pill">{str(hero.get('release_date',''))[:4]}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Trending grid
    st.markdown('<div class="section-title">🔥 Trending This Week</div>', unsafe_allow_html=True)
    render_tmdb_grid(trending, cols=5)

    # Top rated
    st.markdown('<div class="section-title">⭐ Top Rated All Time</div>', unsafe_allow_html=True)
    with st.spinner("Loading top rated..."):
        top_rated = get_top_rated()
    render_tmdb_grid(top_rated, cols=5)


# ─────────────────────────────────────────────
# PAGE: SEARCH & RECOMMEND
# ─────────────────────────────────────────────
with tab2:
    st.markdown("""
    <div style="margin-bottom:20px;">
        <div style="font-size:0.85rem; color:#888;">
            Powered by hybrid TF-IDF + rating signals
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_sel, col_fil = st.columns([2, 1])
    with col_sel:
        selected = st.selectbox(
            "Pick a movie you love",
            options=["— Select a movie —"] + sorted(df["title"].tolist()),
        )
    with col_fil:
        genre_filter = st.multiselect(
            "Filter by genre",
            options=["Action","Drama","Sci-Fi","Crime","Thriller","Comedy","Romance","Horror","Adventure"],
        )

    if selected and selected != "— Select a movie —":
        recs, top_genre = get_recommendations(selected, df, cosine_sim)

        # Apply genre filter
        if genre_filter and not recs.empty:
            recs = recs[recs["genres"].apply(
                lambda g: any(f.lower() in g.lower() for f in genre_filter)
            )]

        if not recs.empty:
            st.markdown(f"""
            <div style="margin: 20px 0 8px 0;">
                <span style="color:#888; font-size:0.85rem;">Because you liked</span>
                <span style="font-weight:700; color:#e50914; font-size:1rem;"> {selected}</span>
                <span style="color:#888; font-size:0.85rem;">, you might enjoy:</span>
            </div>
            """, unsafe_allow_html=True)

            render_local_grid(recs, cols=4)
        else:
            st.info("No recommendations match your filters. Try removing some genre filters.")

    # Movie detail modal (when Details button clicked)
    if st.session_state.selected_movie:
        mid = st.session_state.selected_movie
        with st.spinner("Loading details..."):
            details = get_movie_details(mid)
            providers = get_watch_providers(mid)

        if details:
            st.markdown("---")
            c1, c2 = st.columns([1, 2])
            with c1:
                st.image(poster_url(details.get("poster_path")), use_container_width=True)
            with c2:
                st.markdown(f"""
                <div style="padding: 10px 0;">
                    <div style="font-family: 'Bebas Neue'; font-size:2rem; letter-spacing:2px; color:#fff;">
                        {details.get('title','')}
                    </div>
                    <div style="color:#888; font-size:0.85rem; margin: 6px 0 14px;">
                        {details.get('release_date','')[:4]} &nbsp;•&nbsp;
                        {details.get('runtime',0)} min &nbsp;•&nbsp;
                        {', '.join([g['name'] for g in details.get('genres',[])])}
                    </div>
                    <div style="font-size:0.9rem; color:#ccc; line-height:1.7; margin-bottom:16px;">
                        {details.get('overview','')}
                    </div>
                    <span class="rating-badge" style="font-size:0.9rem; padding:4px 12px;">
                        ★ {details.get('vote_average',0):.1f} / 10
                    </span>
                </div>
                """, unsafe_allow_html=True)

                # Trailer
                videos = details.get("videos", {}).get("results", [])
                trailer = next((v for v in videos if v["type"] == "Trailer" and v["site"] == "YouTube"), None)
                if trailer:
                    st.markdown(f"[▶ Watch Trailer on YouTube](https://youtube.com/watch?v={trailer['key']})")

                # Watch providers
                if providers.get("flatrate"):
                    names = [p["provider_name"] for p in providers["flatrate"][:4]]
                    st.markdown(f"**Where to Watch:** {' · '.join(names)}")

                if st.button("✕ Close Details"):
                    st.session_state.selected_movie = None
                    st.rerun()


# ─────────────────────────────────────────────
# PAGE: WATCHLIST
# ─────────────────────────────────────────────
with tab3:
    if not st.session_state.watchlist:
        st.markdown("""
        <div style="text-align:center; padding: 80px 0; color:#555;">
            <div style="font-size:3rem; margin-bottom:16px;">🎬</div>
            <div style="font-size:1.1rem;">Your watchlist is empty.</div>
            <div style="font-size:0.85rem; margin-top:8px;">
                Go to <b>Search & Recommend</b> and add movies you want to watch.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        cols = st.columns(4)
        for i, title in enumerate(st.session_state.watchlist):
            with cols[i % 4]:
                results = search_tmdb(title)
                p = poster_url(results[0].get("poster_path") if results else None)
                rating = results[0].get("vote_average", 0) if results else 0
                year = str(results[0].get("release_date",""))[:4] if results else ""

                st.markdown(f"""
                <div class="movie-card">
                    <img src="{p}" alt="{title}" loading="lazy"/>
                    <div class="movie-card-body">
                        <div class="movie-card-title">{title}</div>
                        <div class="movie-card-meta">
                            <span class="rating-badge">★ {rating:.1f}</span>
                            <span>{year}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if st.button("🗑 Remove", key=f"rm_{i}"):
                    st.session_state.watchlist.remove(title)
                    st.rerun()

        st.markdown("---")
        if st.button("🗑 Clear Entire Watchlist"):
            st.session_state.watchlist = []
            st.rerun()
