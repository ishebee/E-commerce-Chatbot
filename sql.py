from groq import Groq
import os
import re
import sqlite3
import pandas as pd
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Load API Key from Streamlit Secrets
GROQ_MODEL = st.secrets["GROQ_MODEL"]

db_path = Path(__file__).parent / "db.sqlite"

client_sql = Groq(api_key=GROQ_MODEL)

sql_prompt = """You are an expert in understanding the database schema and generating SQL queries for a natural language question asked
pertaining to the data you have. The schema is provided in the schema tags. 
<schema> 
table: product 

fields: 
product_link - string (hyperlink to product)    
title - string (name of the product)    
brand - string (brand of the product)    
price - integer (price of the product in Indian Rupees)    
discount - float (discount on the product. 10 percent discount is represented as 0.1, 20 percent as 0.2, and such.)    
avg_rating - float (average rating of the product. Range 0-5, 5 is the highest.)    
total_ratings - integer (total number of ratings for the product)

</schema>
Make sure whenever you try to search for the brand name, the name can be in any case. 
So, make sure to use %LIKE% to find the brand in condition. Never use "ILIKE". 
Create a single SQL query for the question provided. 
The query should have all the fields in SELECT clause (i.e. SELECT *)

Just the SQL query is needed, nothing more. Always provide the SQL in between the <SQL></SQL> tags."""

def generate_sql_query(question):
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
    if query.strip().upper().startswith('SELECT'):
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(query, conn)
            return df

def sql_chain(question):
    sql_query = generate_sql_query(question)
    pattern = "<SQL>(.*?)</SQL>"
    matches = re.findall(pattern, sql_query, re.DOTALL)

    if len(matches) == 0:
        return "Sorry, LLM is not able to generate a query for your question"

    print(matches[0].strip())

    response = run_query(matches[0].strip())
    if response is None:
        return "Sorry, there was a problem executing SQL query"

    return response.to_dict(orient='records')
