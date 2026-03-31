import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure Streamlit page
st.set_page_config(page_title="Manga Recommender System", layout="wide")

# Load and clean data with caching to optimize performance
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv('manga.csv')
    # Choose relevant columns and reorder
    cols = ['Title', 'Score', 'Vote', 'Ranked', 'Members', 'Genres', 'Themes', 'Author', 'Demographics']
    df = df[cols]
    
    # Clean Members column: Remove commas and convert to numeric
    df['Members'] = df['Members'].astype(str).str.replace(',', '', regex=False).str.strip()
    df['Members'] = pd.to_numeric(df['Members'], errors='coerce').fillna(0).astype(int)
    
    # Clean data in text columns: Remove brackets, quotes, convert to lowercase, and strip whitespace
    text_cols = ['Genres', 'Themes', 'Author', 'Demographics']
    for col in text_cols:
        df[col] = df[col].astype(str).replace(['[]', 'nan', 'None'], '', regex=False)
        df[col] = df[col].str.replace(r"[\[\]']", "", regex=True).str.lower().str.strip()
    
    # Create a combined "soup" column for content-based filtering
    df['soup'] = (df['Genres'] + ' ') * 2 + df['Themes'] + ' ' + df['Author'] + ' ' + df['Demographics']
    df = df.dropna(subset=['Title']).reset_index(drop=True)
    return df

# Model building with caching to avoid recomputation
@st.cache_resource
def build_model(_df):
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(_df['soup'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(_df.index, index=_df['Title']).drop_duplicates()
    return tfidf_matrix, cosine_sim, indices

# Load data and build model
df = load_and_clean_data()
tfidf_matrix, cosine_sim, indices = build_model(df)

# UI with Streamlit
st.title("📚 Manga Recommendation System")
st.markdown("Khám phá những bộ Manga tương đồng dựa trên sở thích của bạn!")

# Sidebar for user input
st.sidebar.header("Tùy chọn gợi ý")
selected_manga = st.sidebar.selectbox("Chọn bộ Manga bạn thích:", df['Title'].values)
num_rec = st.sidebar.slider("Số lượng gợi ý:", 5, 20, 10)

if st.button('Tìm kiếm gợi ý'):
    # Index of the selected manga
    idx = indices[selected_manga]
    if hasattr(idx, "__len__"): idx = idx[0]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Take the top 50 similar manga (excluding the selected one)
    top_indices = [i[0] for i in sim_scores[1:51]]
    temp_df = df.iloc[top_indices].copy()
    recs = temp_df.sort_values(by=['Score', 'Members'], ascending=False).head(num_rec)
    
    # Show recommendations
    st.subheader(f"Các bộ truyện tương tự như: {selected_manga}")
    
    cols = st.columns(2)
    for i, (index, row) in enumerate(recs.iterrows()):
        with cols[i % 2]:
            with st.container():
                st.write(f"### {i+1}. {row['Title']}")
                st.write(f"⭐ **Score:** {row['Score']} | 👥 **Members:** {row['Members']}")
                st.write(f"🏷️ **Genres:** {row['Genres']}")
                st.write(f"✍️ **Author:** {row['Author']}")
                st.divider()