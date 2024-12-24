import os
from fastapi import FastAPI
from pydantic import BaseModel
import openai
import psycopg2
from dotenv import load_dotenv
import numpy as np
from scipy.spatial.distance import cosine

# Load environment variables from .env
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize FastAPI
app = FastAPI()

# Pydantic model for request body
class QueryRequest(BaseModel):
    query: str

# Connect to PostgreSQL vector database
def get_db_connection():
    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST'),
        database=os.getenv('POSTGRES_DB'),
        user=os.getenv('POSTGRES_USER'),
        password=os.getenv('POSTGRES_PASSWORD')
    )
    return conn

# Generate embedding for text using OpenAI
def generate_embedding(text: str):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return np.array(response['data'][0]['embedding'])

# Perform similarity search in the vector database
def similarity_search(query_embedding, top_k=3):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Query the vector database for the most similar vectors
    cursor.execute("""
        SELECT content, embedding FROM vectors
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    # Calculate cosine similarity for each row and return the top_k most similar
    similarities = []
    for row in rows:
        content, embedding = row
        embedding = np.array(embedding)  # Convert to numpy array
        similarity = 1 - cosine(query_embedding, embedding)
        similarities.append((content, similarity))
    
    # Sort by similarity and get top_k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# Generate a response using OpenAI, based on the retrieved context
def generate_openai_response(query, context):
    prompt = f"User query: {query}\n\nRelevant information from the database:\n{context}\n\nAnswer:"
    response = openai.Completion.create(
        model="text-davinci-003",  # You can use other models, but Davinci is a strong one for completion
        prompt=prompt,
        max_tokens=150
    )
    return response['choices'][0]['text'].strip()

# API endpoint to handle user query
@app.post("/query/")
async def handle_query(request: QueryRequest):
    query = request.query
    try:
        # Step 1: Generate embedding for the query
        query_embedding = generate_embedding(query)

        # Step 2: Perform similarity search in the database
        retrieved_data = similarity_search(query_embedding)

        # If no results are found
        if not retrieved_data:
            return {"query": query, "response": "I can not find a proper answer, the input query is not valid."}

        # Step 3: Construct context from the retrieved data
        context = " ".join([content for content, _ in retrieved_data])

        # Step 4: Generate a response using OpenAI based on the context from the database
        response_text = generate_openai_response(query, context)

        # Respond with the generated answer
        return {"query": query, "response": response_text}

    except Exception as e:
        return {"error": str(e)}

# Run the application: `uvicorn app:app --host 0.0.0.0 --port 8000`
