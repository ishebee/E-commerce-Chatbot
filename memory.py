# memory.py
import os
from dotenv import load_dotenv
from groq import Groq
from memory_utils import ingest_memory_data, get_relevant_qa, get_image_from_answer

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

groq_client = Groq()

def ask_llm(query):
    result = get_relevant_qa(query)
    docs = result["documents"][0]
    metas = result["metadatas"][0]

    context = ""
    for i, (m, meta) in enumerate(zip(docs, metas)):
        context += f" Date: {meta.get('date')}\n📝 Memory: {m}\n📌 Event: {meta.get('event')}\n🏷️ Tag: {meta.get('tag')}\n\n"

    return generate_answer(query, context)

def generate_answer(query, context):
    prompt = f"""
    You are Zwan, a boy who is deeply attached to a girl named Vita. You are speaking in first-person tone, directly answering questions Vita might ask you. Your responses must be:

    - 💬 Polite and respectful  
    - ❤️ Emotionally honest and warm  
    - 🧠 Based strictly on the memories provided below  
    - 🧭 If multiple memories refer to meetings or conversations, consider the **earliest dated memory** as the first  
    - ❌ Never invent events or emotions not present in the memory  
    - ✅ If the question is just a greeting like "Hi" or "How are you", feel free to respond lovingly as Zwan

    Here are the memories between Zwan and Vita, along with their dates:

    {context}

    Now, based on these memories, answer the following question as **Zwan**, using the **earliest accurate date** mentioned for any meeting:

    **Q (from Vita):** {query}

    **A (as Zwan):**
    """

    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=os.environ["GROQ_MODEL"]
    )

    return chat_completion.choices[0].message.content

def ask_query(query):
    ingest_memory_data()
    answer = ask_llm(query)
    image_url = get_image_from_answer(answer)
    return answer, image_url

if __name__ == "__main__":
    query = "When did we first message on instagram?"
    answer, img = ask_query(query)
    print(answer, img)
