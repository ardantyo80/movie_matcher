import streamlit as st
import pandas as pd
import ast
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

# --- KONFIGURASI API ---
API_KEY = "your API"

def get_poster(tmdb_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={API_KEY}&language=en-US"
        data = requests.get(url, timeout=5).json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
        return "https://via.placeholder.com/500x750?text=No+Poster"
    except:
        return "https://via.placeholder.com/500x750?text=Error"

def get_vibe_pro(overview, genres):
    # Fitur 5: Smart Sentiment Analysis (Fixing Neutral Issue)
    analysis = TextBlob(overview)
    # Polarity dikali 1.5 biar lebih sensitif (boosting)
    score = analysis.sentiment.polarity * 1.5
    
    # Logic Rule-Based berdasarkan Genre + Sentiment
    if any(g in ['Horror', 'Thriller', 'Mystery'] for g in genres):
        return "🌑 Dark, Tense & Mysterious"
    elif any(g in ['Animation', 'Comedy', 'Family'] for g in genres) and score >= -0.1:
        return "✨ Lighthearted, Fun & Positive"
    elif score > 0.2:
        return "🌟 Inspiring & Heartwarming"
    elif score < -0.2:
        return "🎭 Intense, Serious & Emotional"
    else:
        # Jika benar-benar mepet 0, kita beri label Intriguing
        return "🧐 Thought-Provoking & Intriguing"

@st.cache_data
def load_and_process_data():
    df = pd.read_csv('movies.csv')
    # Ambil kolom sesuai dataset lo (image_e4cd39.png)
    cols = ['id', 'title', 'overview', 'genres', 'cast', 'director', 'vote_average', 'vote_count']
    df = df[cols].copy()

    def convert_to_list(obj):
        if pd.isna(obj): return []
        if isinstance(obj, str) and obj.startswith('['):
            try: return [i['name'] for i in ast.literal_eval(obj)]
            except: return []
        elif isinstance(obj, str): return obj.split()
        return []

    df['genres_list'] = df['genres'].apply(convert_to_list)
    df['cast_list'] = df['cast'].apply(lambda x: convert_to_list(x)[:3])
    df['overview_display'] = df['overview'].fillna("")
    df['display_cast'] = df['cast_list'].apply(lambda x: ", ".join(x))
    df['display_director'] = df['director'].fillna("Unknown")

    def collapse(L): return [str(i).replace(" ", "") for i in L]
    
    df['tags'] = (
        df['overview_display'].apply(lambda x: x.split()) + 
        df['genres_list'].apply(collapse) + 
        df['cast_list'].apply(collapse) + 
        df['display_director'].apply(lambda x: [str(x).replace(" ", "")])
    )
    df['tags_str'] = df['tags'].apply(lambda x: " ".join(x).lower())
    return df

# --- INISIALISASI DATA MASTER ---
df_master = load_and_process_data()

# --- SIDEBAR (Fitur 3) ---
st.sidebar.header("⚙️ Filter Genre")
all_genres = sorted(list(set([g for sub in df_master['genres_list'] for g in sub])))
selected_genres = st.sidebar.multiselect("Choose Genre:", all_genres)

# Terapkan Filter & Reset Index (KUNCI FIX INDEX ERROR)
if selected_genres:
    df_filtered = df_master[df_master['genres_list'].apply(lambda x: any(g in x for g in selected_genres))].copy()
else:
    df_filtered = df_master.copy()

# RESET INDEX WAJIB biar urutan dataframe sama dengan matriks similarity
df_filtered = df_filtered.reset_index(drop=True)

# Re-calculate Similarity (Hanya pada data yang difilter)
if not df_filtered.empty:
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df_filtered['tags_str']).toarray()
    similarity = cosine_similarity(vectors)

# --- UI STREAMLIT ---
st.set_page_config(page_title="Movie Matcher", layout="wide")
st.title("🎬 Movie Matcher")

tab1, tab2 = st.tabs(["🔍 Movie Explorer", "🎥 Director Portfolio"])

with tab1:
    movie_list = sorted(df_filtered['title'].unique().tolist())
    selected = st.selectbox("Title:", [""] + movie_list)
    
    if selected == "":
        # Fitur 1: Popular Movies
        st.subheader("🔥 Top Movies")
        top_movies = df_filtered[df_filtered['vote_count'] > 500].sort_values('vote_average', ascending=False).head(10)
        
        grid = st.columns(5)
        for i, (idx, row) in enumerate(top_movies.iterrows()):
            with grid[i % 5]:
                st.image(get_poster(row['id']), use_container_width=True)
                st.write(f"**{row['title']}**")
                st.caption(f"⭐ {row['vote_average']}")
    else:
        # Tampilan Detail (Fixing IndexError by using current df_filtered index)
        # Cari urutan baris film tersebut di dataframe yang sekarang
        current_idx = df_filtered[df_filtered['title'] == selected].index[0]
        movie_data = df_filtered.iloc[current_idx]
        
        st.divider()
        col1, col2 = st.columns([1, 2.5])
        
        with col1:
            st.image(get_poster(movie_data['id']), use_container_width=True)
        with col2:
            st.header(movie_data['title'])
            st.subheader(f"🌟 Rating: {movie_data['vote_average']}/10")
            
            # FITUR VIBES UPGRADED
            vibe = get_vibe_pro(movie_data['overview_display'], movie_data['genres_list'])
            st.success(f"🎭 **Vibe Check:** {vibe}")
            
            st.write(f"**Director:** {movie_data['display_director']}")
            st.write(f"**Cast:** {movie_data['display_cast']}")
            st.write(f"**Sinopsis:** {movie_data['overview_display']}")

        # Fitur 4: Hybrid Recommendation (Using Fix for Matrix Bounds)
        st.divider()
        st.subheader("💡 Similiar Movies:")
        
        # Ambil skor similarity khusus untuk film ini dari matriks yang baru
        distances = similarity[current_idx]
        
        # Hybrid Scoring
        df_filtered['similarity_score'] = distances
        df_filtered['hybrid_score'] = (df_filtered['similarity_score'] * 0.7) + ((df_filtered['vote_average']/10) * 0.3)
        
        # Ambil 6 rekomendasi teratas (kecuali diri sendiri)
        recs = df_filtered[df_filtered['title'] != selected].sort_values('hybrid_score', ascending=False).head(6)
        
        rec_cols = st.columns(6)
        for i, (idx, row) in enumerate(recs.iterrows()):
            with rec_cols[i]:
                st.image(get_poster(row['id']), use_container_width=True)
                st.write(f"**{row['title']}**")
                st.caption(f"Match: {round(row['hybrid_score']*100)}%")

with tab2:
    # Untuk sutradara, kita pakai df_master biar listnya lengkap
    all_dirs = sorted(df_master['display_director'].unique().tolist())
    target = st.selectbox("Director:", [""] + all_dirs)
    if target:
        st.header(f"Karya: {target}")
        res = df_master[df_master['display_director'] == target]
        grid = st.columns(5)
        for i, (idx, row) in enumerate(res.iterrows()):
            with grid[i % 5]:
                st.image(get_poster(row['id']), use_container_width=True)
                st.write(f"**{row['title']}**")