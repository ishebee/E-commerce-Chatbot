import sys
import os
import re
import sqlite3
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# ✅ Ensure SQLite is correctly overridden
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")  # Force using updated SQLite
except ImportError:
    print("⚠️ pysqlite3-binary is missing. Install it using `pip install pysqlite3-binary`.")

# ✅ Load API key from Streamlit Secrets
import streamlit as st
GROQ_MODEL = st.secrets.get("GROQ_MODEL")

if not GROQ_MODEL:
    st.error("❌ GROQ_MODEL is missing in Streamlit Secrets! Set it in 'secrets.toml'.")

db_path = Path(__file__).parent / "db.sqlite"

client_sql = Groq()

# SQL Query Generation Prompt
sql_prompt = """You are an expert in SQL query generation. Generate a SQL query using the schema:
<schema>
table: product
fields:
- product_link (string)
- title (string)
- brand (string)
- price (integer)
- discount (float)
- avg_rating (float)
- total_ratings (integer)
</schema>
Use 'LIKE' for brand searches. Do NOT use 'ILIKE'. Wrap the SQL query in <SQL>...</SQL> tags.
"""

# Comprehension Prompt
comprehension_prompt = """Analyze the given dataset and generate a human-readable response based on the provided data. 
Reply in a structured format with product title, price, discount, rating, and product link.
Example:
1. Campus Women Running Shoes: Rs. 1104 (35% off), Rating: 4.4 <link>
2. Adidas Sports Shoes: Rs. 2500 (20% off), Rating: 4.6 <link>
"""


def generate_sql_query(question):
    """Generates SQL query using Groq API."""
    chat_completion = client_sql.chat.completions.create(
        messages=[
            {"role": "system", "content": sql_prompt},
            {"role": "user", "content": question}
        ],
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=1024
    )
    return chat_completion.choices[0].message.content


def run_query(query):
    """Executes SQL query on SQLite database."""
    if query.strip().upper().startswith('SELECT'):
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(query, conn)
            return df


def data_comprehension(question, context):
    """Generates human-readable answer from SQL query result."""
    chat_completion = client_sql.chat.completions.create(
        messages=[
            {"role": "system", "content": comprehension_prompt},
            {"role": "user", "content": f"QUESTION: {question}. DATA: {context}"}
        ],
        model=GROQ_MODEL,
        temperature=0.2
    )
    return chat_completion.choices[0].message.content


def sql_chain(question):
    """Handles full SQL pipeline from query generation to execution & response generation."""
    sql_query = generate_sql_query(question)
    pattern = "<SQL>(.*?)</SQL>"
    matches = re.findall(pattern, sql_query, re.DOTALL)

    if not matches:
        return "Sorry, I couldn't generate a SQL query for your question."

    response = run_query(matches[0].strip())

    if response is None:
        return "Sorry, there was an issue executing the SQL query."

    context = response.to_dict(orient='records')
    return data_comprehension(question, context)


# ✅ Ensure function is properly exported
__all__ = ["sql_chain"]
