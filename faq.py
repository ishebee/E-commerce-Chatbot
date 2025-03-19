import os
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load API Key from Streamlit Secrets
GROQ_MODEL = st.secrets["GROQ_MODEL"]

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)
faqs_path = Path(__file__).parent / "resources/faq_data.csv"
chroma_client = chromadb.Client()
groq_client = Groq(api_key=GROQ_MODEL)
collection_name_faq = 'faqs'

def ingest_faq_data(path):
    if collection_name_faq not in [c.name for c in chroma_client.list_collections()]:
        print("Ingesting FAQ data into Chromadb...")
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
        print(f"FAQ Data successfully ingested into Chroma collection: {collection_name_faq}")
    else:
        print(f"Collection: {collection_name_faq} already exists")

def generate_answer(query, context):
    prompt = f'''Given the following context and question, generate answer based on this context only.
    If the answer is not found in the context, kindly state "I don't know". Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {query}
    '''
    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {'role': 'user', 'content': prompt}
        ]
    )
    return completion.choices[0].message.content
