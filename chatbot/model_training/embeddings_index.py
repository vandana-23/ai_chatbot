from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Correct paths using forward slashes
DATA_FILE = BASE_DIR / "data/company_data.txt"
INDEX_DIR = BASE_DIR / "faiss_index"  
INDEX_DIR.mkdir(parents=True, exist_ok=True)  

INDEX_FILE = INDEX_DIR / "goklyn_index.faiss"
TEXTS_FILE = INDEX_DIR / "texts.npy"

model = SentenceTransformer("all-MiniLM-L6-v2")

# Read corpus
with open(DATA_FILE, "r", encoding="utf-8") as f:
    corpus = [line.strip() for line in f if line.strip()]

# Generate embeddings
embeddings = model.encode(corpus, convert_to_numpy=True, batch_size=32)

# Build FAISS index
faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
faiss_index.add(embeddings)

# Save index and corpus
faiss.write_index(faiss_index, str(INDEX_DIR / "goklyn_index.faiss"))
np.save(str(INDEX_DIR / "texts.npy"), corpus)

print("Embedding index built and saved!")

def load_faiss_index():
    index = faiss.read_index(str(INDEX_DIR / "goklyn_index.faiss"))
    texts = np.load(str(INDEX_DIR / "texts.npy"), allow_pickle=True)
    return index, texts