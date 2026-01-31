import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

# --- PATHS (Colab Default) ---
EMBEDS_DIR = '/content/drive/MyDrive/HackathonProject/embeddings'

# --- LOAD RESOURCES (Cached) ---
@st.cache_resource
def load_resources():
    param_path = os.path.join(EMBEDS_DIR, "video_embeddings.npy")
    
    if not os.path.exists(param_path):
        return None, None, None

    # Load Embeddings
    v_embs = np.load(os.path.join(EMBEDS_DIR, "video_embeddings.npy"))
    v_paths = np.load(os.path.join(EMBEDS_DIR, "video_paths.npy"))
    
    # Init FAISS
    dimension = v_embs.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(v_embs)
    index.add(v_embs)
    
    # Load Model
    model = SentenceTransformer('clip-ViT-B-32')
    return index, v_paths, model

index, video_paths, model = load_resources()

# --- UI ---
st.set_page_config(page_title="Semantic Footage Search", page_icon="🎬", layout="wide")

st.title("🎬 Semantic Footage Search")
st.markdown("""
**Find the intent behind the clip.**
Enter a query like _"tense pause before dialogue"_ or _"happy reunion at airport"_.
""")

if index is None:
    st.warning("⚠️ **Embeddings not found.**")
    st.info("Please wait for Member 2 to generate `video_embeddings.npy` in `/content/drive/MyDrive/HackathonProject/embeddings`.")
    st.stop()

query = st.text_input("🔍 Search Query", "", placeholder="Describe the scene...")

if query:
    with st.spinner("Analyzing semantic meaning..."):
        # Search Logic
        text_emb = model.encode([query])
        faiss.normalize_L2(text_emb)
        D, I = index.search(text_emb, k=6)
        
        st.subheader(f"Top Matches for: _'{query}'_")
        
        cols = st.columns(3)
        for i, idx in enumerate(I[0]):
            file_path = video_paths[idx]
            match_score = D[0][i]
            
            # Display in Grid
            with cols[i % 3]:
                st.video(file_path)
                st.caption(f"**Score:** {match_score:.2f} | `{os.path.basename(file_path)}`")
