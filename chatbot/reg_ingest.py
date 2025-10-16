
import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings

BASE = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE, "vector_db")
Path(DB_DIR).mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2" 

def build_vector_db(file_path):
    print(f"Loading file: {file_path}")
    loader = TextLoader(file_path, encoding='utf-8')
    docs = loader.load()

    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    print("Building vector database...")
    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma.from_documents(
        chunks,
        embedding,
        persist_directory=DB_DIR
    )
    
    print(" Vector DB successfully created and saved.")

if __name__ == "__main__":
   
    file_path = os.path.join(BASE, "./data/company_data.txt")
    build_vector_db(file_path)
