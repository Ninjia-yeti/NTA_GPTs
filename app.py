import os
import psycopg2
import openai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import numpy as np
from scipy.spatial.distance import cosine



# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables for database and OpenAI API
DB_HOST = os.getenv('POSTGRES_HOST','localhost')
DB_NAME = os.getenv('POSTGRES_NAME','nta_data')
DB_USER = os.getenv('POSTGRES_USER','postgres')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD','nta_gpts')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI API key
openai.api_key = OPENAI_API_KEY

# FastAPI app initialization
app = FastAPI()

def connect_to_db():
    """Create a connection to the PostgreSQL database using environment variables."""
    print("Connecting to the DB...")
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

# def generate_embedding(text):
#     """Generate embeddings for the given text using OpenAI API."""
#     print(f"Generating embedding for query: {text}")
#     response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
#     embedding = response['data'][0]['embedding']
#     print(f"Query embedding: {embedding[:10]}...")  # Log first 10 values of the embedding for brevity
#     return embedding

def generate_embedding(text):
    """Generate embeddings for the given text using OpenAI API, ensuring exactly 1536-dimensional embeddings."""
    print(f"Generating embedding for query: {text}")
    
    # Call OpenAI API to generate the embedding
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    embedding = response['data'][0]['embedding']
    
    # Ensure the embedding has the correct dimension (1536)
    embedding_dimension = 1536
    current_dimension = len(embedding)
    
    if current_dimension < embedding_dimension:
        # Pad with zeros if embedding is smaller than 1536 dimensions
        padding = [0] * (embedding_dimension - current_dimension)
        embedding.extend(padding)
        print(f"Padding applied. New embedding dimension: {len(embedding)}")
    
    elif current_dimension > embedding_dimension:
        # Truncate the embedding to 1536 dimensions if it's too large
        embedding = embedding[:embedding_dimension]
        print(f"Truncating applied. New embedding dimension: {len(embedding)}")
    
    # Log the first 10 values for debugging
    print(f"Query embedding (first 10 values): {embedding[:10]}...")

    return embedding




# def similarity_search(query_embedding):
#     """Perform similarity search on the database to find the closest embeddings."""
#     conn = connect_to_db()
#     print("---------------DB connected-----------")
#     if conn:
#         cursor = conn.cursor()
#         cursor.execute("""SELECT content, embedding FROM vectors""")
#         rows = cursor.fetchall()
#         cursor.close()
#         conn.close()

#         max_similarity = float('inf')  # We are looking for the minimum cosine distance
#         best_match = None

#         for row in rows:
#             content, embedding = row
            
#             # Ensure embedding is a 1D numpy array
#             embedding = np.array(embedding)
#             if embedding.ndim > 1:
#                 print("Embedding is not 1D, flattening it!")
#                 embedding = embedding.flatten()  # Ensure it's 1D
#             print(f"Embedding shape: {embedding.shape}")
            
#             # Ensure query_embedding is also 1D
#             query_embedding = np.array(query_embedding)
#             if query_embedding.ndim > 1:
#                 print("Query embedding is not 1D, flattening it!")
#                 query_embedding = query_embedding.flatten()  # Ensure it's 1D
#             print(f"Query embedding shape: {query_embedding.shape}")
            
#             # Now we can safely compute cosine similarity
#             similarity = cosine(query_embedding, embedding)
#             if similarity < max_similarity:  # Smaller distance means more similarity
#                 max_similarity = similarity
#                 best_match = content
        
#         return best_match
#     else:
#         raise HTTPException(status_code=500, detail="Database connection failed")


def ensure_embedding_dimension(embedding, target_dimension=1536):
    """Ensure the embedding has the specified dimension by padding or truncating."""
    current_dimension = len(embedding)
    
    if current_dimension < target_dimension:
        # Pad with zeros if embedding is smaller than the target dimension
        padding = [0] * (target_dimension - current_dimension)
        embedding.extend(padding)
        print(f"Padding applied. New embedding dimension: {len(embedding)}")
    
    elif current_dimension > target_dimension:
        # Truncate the embedding to the target dimension if it's too large
        embedding = embedding[:target_dimension]
        print(f"Truncating applied. New embedding dimension: {len(embedding)}")
    
    return embedding

def similarity_search(query_embedding):
    """Perform similarity search on the database to find the closest embeddings."""
    conn = connect_to_db()
    print("---------------DB connected-----------")
    
    if conn:
        cursor = conn.cursor()
        cursor.execute("""SELECT content, embedding FROM vectors""")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        max_similarity = float('inf')  # We are looking for the minimum cosine distance
        best_match = None

        # Ensure the query_embedding has the correct dimension
        query_embedding = np.array(query_embedding)
        query_embedding = ensure_embedding_dimension(query_embedding.tolist())

        for row in rows:
            content, embedding = row
            
            # Convert the stored embedding to numpy array and ensure it's 1D
            embedding = np.array(embedding)
            embedding = ensure_embedding_dimension(embedding.tolist())

            # Compute cosine similarity between the query and stored embedding
            similarity = cosine(query_embedding, embedding)
            
            if similarity < max_similarity:  # Smaller distance means more similarity
                max_similarity = similarity
                best_match = content
        
        return best_match
    else:
        raise HTTPException(status_code=500, detail="Database connection failed")


def openai_generate_answer(query, context=""):
    """Generate a response from OpenAI's GPT based on the query and context."""
    prompt = f"Answer the following question based on the context: {context}\n\nQuestion: {query}\nAnswer:"
    response = openai.Completion.create(
        model="text-davinci-003",  # You can adjust to the model you prefer
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7
    )
    return response['choices'][0]['text'].strip()

@app.post("/chat/")
async def chat(query: str):
    """Process user input and return the most relevant database result or generate a response from OpenAI."""
    print(f"Received query: {query}")
    
    # Generate embedding for the query
    query_embedding = generate_embedding(query)
    
    # Perform similarity search to find the closest match in the database
    best_match = similarity_search(query_embedding)
    
    if best_match:
        # If a match is found, use that as context for OpenAI's response
        context = best_match
        print(f"Found context: {context[:30]}...")
    else:
        # If no match is found, use a default response
        context = ""
        print("No relevant context found in the database.")
    
    # Generate OpenAI's response using the query and the best match context (if any)
    answer = openai_generate_answer(query, context)
    
    print(f"Generated answer: {answer}")
    
    return {"answer": answer}

# Main execution (if needed, use for testing or debugging)
if __name__ == "__main__":
    import uvicorn
    
