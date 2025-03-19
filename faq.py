import os
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
import pandas
import streamlit as st

# ✅ Load API key from Streamlit Secrets
GROQ_MODEL = st.secrets.get("GROQ_MODEL")

if not GROQ_MODEL:
    st.error("❌ GROQ_MODEL is missing in Streamlit Secrets! Set it in 'secrets.toml'.")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)
faqs_path = Path(__file__).parent / "resources/faq_data.csv"
chroma_client = chromadb.Client()
groq_client = Groq()
collection_name_faq = 'faqs'


def ingest_faq_data(path):
    if collection_name_faq not in [c.name for c in chroma_client.list_collections()]:
        print("Ingesting FAQ data into Chromadb...")
        collection = chroma_client.create_collection(
            name=collection_name_faq,
            embedding_function=ef
        )
        df = pandas.read_csv(path)
        docs = df['question'].to_list()
        metadata = [{'answer': ans} for ans in df['answer'].to_list()]
        ids = [f"id_{i}" for i in range(len(docs))]
        collection.add(
            documents=docs,
            metadatas=metadata,
            ids=ids
        )
        print(f"FAQ Data successfully ingested into Chroma collection: {collection_name_faq}")
    else:
        print(f"Collection: {collection_name_faq} already exist")
