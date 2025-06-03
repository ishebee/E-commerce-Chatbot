import sys
import os
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")  # Force updated SQLite
except ImportError:
    print("⚠️ pysqlite3-binary is missing. Install it using `pip install pysqlite3-binary`.")

from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
_ = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# Define embedding function
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

# ✅ Use **EphemeralClient** instead of Client (Fix for Streamlit Cloud)
chroma_client = chromadb.EphemeralClient()

# Collection name
collection_name_faq = 'faqs'
faqs_path = Path(__file__).parent / "resources/faq_data.csv"


def ingest_faq_data(path):
    """Ensure the FAQ collection exists, reloading if necessary (Fix for Chroma v0.6.0)"""
    print("Checking FAQ data in ChromaDB...")

    try:
        collection = chroma_client.get_collection(collection_name_faq)
        print(f"✅ Collection `{collection_name_faq}` already exists.")
    except Exception:
        print(f"⚠️ Collection `{collection_name_faq}` not found. Creating new collection...")
        collection = chroma_client.create_collection(
            name=collection_name_faq,
            embedding_function=ef
        )

        # Load FAQs from CSV
        df = pd.read_csv(path)
        docs = df['question'].tolist()
        metadata = [{'answer': ans} for ans in df['answer'].tolist()]
        ids = [f"id_{i}" for i in range(len(docs))]

        # Add FAQs to ChromaDB
        collection.add(
            documents=docs,
            metadatas=metadata,
            ids=ids
        )

        print(f"✅ FAQ Data successfully loaded into ChromaDB: `{collection_name_faq}`")


# ✅ Ensure FAQ data is loaded at startup
ingest_faq_data(faqs_path)


def get_relevant_qa(query):
    """Retrieve relevant answers from ChromaDB."""
    collection = chroma_client.get_collection(name=collection_name_faq)
    result = collection.query(
        query_texts=[query],
        n_results=2
    )
    return result


def faq_chain(query):
    """Retrieve and return the most relevant answer from FAQs."""
    result = get_relevant_qa(query)
    if not result or not result['metadatas'][0]:
        return "I don't know the answer."

    context = "".join([r.get('answer') for r in result['metadatas'][0]])
    return context
