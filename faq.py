import sys
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")  # ✅ Override SQLite
except ImportError:
    print("⚠️ pysqlite3-binary is missing. Install it using `pip install pysqlite3-binary`.")
import os
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define embedding function
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

faqs_path = Path(__file__).parent / "resources/faq_data.csv"
chroma_client = chromadb.PersistentClient(path="faqs_db")  # Make sure it's persisted properly
groq_client = Groq()
collection_name_faq = 'faqs'


def ingest_faq_data(path):
    """Ensure FAQ data is ingested into ChromaDB"""
    collection_names = [c.name for c in chroma_client.list_collections()]

    if collection_name_faq not in collection_names:
        print("Ingesting FAQ data into ChromaDB...")
        collection = chroma_client.create_collection(
            name=collection_name_faq,
            embedding_function=ef
        )
        df = pd.read_csv(path)
        docs = df['question'].to_list()
        metadata = [{'answer': ans} for ans in df['answer'].to_list()]
        ids = [f"id_{i}" for i in range(len(docs))]
        collection.add(
            documents=docs,
            metadatas=metadata,
            ids=ids
        )
        print(f"✅ FAQ Data successfully ingested into ChromaDB collection: {collection_name_faq}")
    else:
        print(f"✅ Collection '{collection_name_faq}' already exists.")


def get_relevant_qa(query):
    """Fetch relevant QA from ChromaDB"""
    collection = chroma_client.get_collection(
        name=collection_name_faq,
        embedding_function=ef
    )
    result = collection.query(
        query_texts=[query],
        n_results=2
    )
    return result


def generate_answer(query, context):
    """Use Groq LLM to generate answer based on context"""
    prompt = f'''Given the following context and question, generate an answer based on this context only.
    If the answer is not found in the context, kindly state "I don't know". Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {query}
    '''
    completion = groq_client.chat.completions.create(
        model=os.environ['GROQ_MODEL'],
        messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ]
    )
    return completion.choices[0].message.content


def faq_chain(query):
    """Query the FAQ system and return the answer"""
    result = get_relevant_qa(query)
    if not result["documents"]:
        return "I don't know."
    
    context = "".join([r.get('answer', '') for r in result['metadatas'][0]])
    answer = generate_answer(query, context)
    return answer


# ✅ Ensure functions are properly exported
__all__ = ["ingest_faq_data", "faq_chain"]
