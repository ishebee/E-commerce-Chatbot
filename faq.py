import os
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
import pandas as pd
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ✅ Load API key from Streamlit Secrets
GROQ_MODEL = st.secrets.get("GROQ_MODEL")

if not GROQ_MODEL:
    st.error("❌ GROQ_MODEL is missing in Streamlit Secrets! Set it in 'secrets.toml'.")

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

faqs_path = Path(__file__).parent / "resources/faq_data.csv"
chroma_client = chromadb.PersistentClient(path="faqs_db")  # Persist FAQ collection
groq_client = Groq()
collection_name_faq = 'faqs'


def ingest_faq_data(path):
    """Ensures FAQ data is ingested into ChromaDB."""
    collection_names = [c.name for c in chroma_client.list_collections()]
    if collection_name_faq not in collection_names:
        collection = chroma_client.create_collection(
            name=collection_name_faq, embedding_function=ef
        )
        df = pd.read_csv(path)
        docs = df['question'].to_list()
        metadata = [{'answer': ans} for ans in df['answer'].to_list()]
        ids = [f"id_{i}" for i in range(len(docs))]
        collection.add(documents=docs, metadatas=metadata, ids=ids)
    else:
        print(f"✅ Collection '{collection_name_faq}' already exists.")


def faq_chain(query):
    """Queries FAQ system and returns an answer."""
    collection = chroma_client.get_collection(
        name=collection_name_faq, embedding_function=ef
    )
    result = collection.query(query_texts=[query], n_results=2)
    if not result["documents"]:
        return "I don't know."

    context = "".join([r.get('answer', '') for r in result['metadatas'][0]])
    return generate_answer(query, context)


# ✅ Ensure functions are properly exported
__all__ = ["ingest_faq_data", "faq_chain"]
