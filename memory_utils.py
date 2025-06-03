# memory_utils.py
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os

mem_path = "resources/Memories.csv"
collection_name = "memories"

chroma_client = chromadb.Client()
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

def ingest_memory_data(path=mem_path):
    if collection_name not in [c.name for c in chroma_client.list_collections()]:
        print(f"Inserting memories into ChromaDB collection: {collection_name}")
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=ef
        )
        df = pd.read_csv(path)
        documents = df["Memory"].tolist()
        metadatas = [
            {
                "date": str(row["Date"]),
                "event": row["Event "],
                "tag": row["Tag"],
                "image_url": row["Image"]
            }
            for _, row in df.iterrows()
        ]
        ids = [f"ids_{i}" for i in range(len(df))]
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        print("Memories inserted.")
    else:
        print("Collection already exists.")

def get_relevant_qa(query, n_results=5):
    collection = chroma_client.get_collection(
        name=collection_name,
        embedding_function=ef
    )
    return collection.query(
        query_texts=[query],
        n_results=n_results
    )

def get_image_from_answer(answer):
    collection = chroma_client.get_collection(
        name=collection_name,
        embedding_function=ef
    )
    match_result = collection.query(
        query_texts=[answer],
        n_results=1
    )
    return match_result["metadatas"][0][0].get("image_url", "")
