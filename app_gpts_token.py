import os
import psycopg2
import openai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
import tiktoken  # OpenAI's tokenizer
from sklearn.decomposition import PCA  # type: ignore

# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables for database and OpenAI API
DB_HOST = os.getenv('POSTGRES_HOST', 'localhost')
DB_NAME = os.getenv('POSTGRES_NAME_TOKEN', 'nta_data_token')
DB_USER = os.getenv('POSTGRES_USER', 'postgres')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'nta_gpts')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI API key
openai.api_key = OPENAI_API_KEY

# FastAPI app initialization
app = FastAPI()

# Initialize the tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-3.5 and GPT-4 use the "cl100k_base" encoding

# Define maximum token limit for GPT-4
MAX_TOKENS = 8192
MIN_SIMILARITY = 0.7
top_n = 5

# Initialize PCA for dimensionality reduction
pca = PCA(n_components=512)

# Set up PCA with a dynamic number of components
def get_pca():
    n_components = min(512, min(1000, 1536))  # Ensure n_components is valid
    return PCA(n_components=n_components, svd_solver='randomized')  # Use randomized solver


# Pydantic model for parsing JSON body
class RequestBody(BaseModel):
    query: str

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

def generate_embedding(text):
    """Generate embeddings for the given text using OpenAI API."""
    print(f"Generating embedding for query: {text}")
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    embedding = response['data'][0]['embedding']
    return embedding


def reduce_embedding_dimension(embedding):
    """Reduce the dimensionality of a 1536-dimensional embedding to 512 dimensions."""
    embedding = np.array(embedding).reshape(1, -1)  # Ensure it's a 2D array

    # Fit PCA only if it hasn't been fitted yet
    global pca
    if not hasattr(pca, 'components_'):  # Check if PCA is already fitted
        print("Fitting PCA for dimensionality reduction...")
        # Generate random embeddings to fit PCA
        random_embeddings = np.random.rand(1000, 1536)  # Simulating some sample embeddings to fit PCA
        pca.fit(random_embeddings)  # Pre-fit the PCA with random embeddings

    reduced_embedding = pca.transform(embedding)  # Reduce dimensionality
    return reduced_embedding[0]


def convert_embedding_string(embedding_str):
    """Convert a string representation of an embedding into a NumPy array."""
    try:
        cleaned_str = embedding_str.strip("[]").replace(" ", "")
        embedding_list = [float(i) for i in cleaned_str.split(',')]
        embedding_array = np.array(embedding_list)
        return embedding_array
    except ValueError as e:
        raise ValueError(f"Error converting embedding string to numpy array: {embedding_str}") from e

def get_cosine_similarity(embedding1, embedding2):
    """Compute the cosine similarity between two embeddings."""
    embedding1 = np.array(embedding1).reshape(1, -1)
    embedding2 = np.array(embedding2).reshape(1, -1)
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity[0][0]

def similarity_search(query_embedding, top_n, min_similarity=0.7):
    """Perform similarity search on the database to find the top N closest embeddings above a minimum similarity threshold."""
    conn = connect_to_db()
    print("---------------DB connected-----------")
    
    if conn:
        cursor = conn.cursor()
        cursor.execute("""SELECT content, embedding FROM vectors""")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        similarities = []  # List to store (similarity, content) tuples

        for row in rows:
            content, embedding = row

            # Convert the string embedding from the database to a NumPy array
            if isinstance(embedding, str):
                embedding = convert_embedding_string(embedding)
            
            # Compute cosine similarity between the query and stored embedding
            similarity = get_cosine_similarity(query_embedding, embedding)

            # Append the similarity score and corresponding content to the list if above the threshold
            if similarity >= min_similarity:
                similarities.append((similarity, content))
        
        # Sort by similarity in descending order and return the top N contents
        top_matches = sorted(similarities, key=lambda x: x[0], reverse=True)[:top_n]
        
        return top_matches
    else:
        raise HTTPException(status_code=500, detail="Database connection failed")


def count_tokens(text):
    """Count the number of tokens in a text string."""
    return len(tokenizer.encode(text))

def openai_generate_answer(query, context):
    """Generate a response from OpenAI's GPT based on the query and context."""
    total_tokens = count_tokens(query) + count_tokens(context)
    print(f"Total tokens (query + context): {total_tokens}")

    # If the total token count exceeds the max, trim the context
    if total_tokens > MAX_TOKENS:
        print("Total tokens exceed limit, trimming context...")
        excess_tokens = total_tokens - MAX_TOKENS
        context_tokens = tokenizer.encode(context)
        context = tokenizer.decode(context_tokens[:-excess_tokens])  # Trim the excess tokens

    if context:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},  # Optional system message
            {"role": "user", "content": f"Answer the following question in Japanese based on the context: {context}\n\nQuestion: {query}\nAnswer:"}
        ]
    else:
        # If no context, provide a general response
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},  # Optional system message
            {"role": "user", "content": f"Answer the following general question in Japanese: {query}\nAnswer:"}
        ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Or "gpt-3.5-turbo" depending on your requirements
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return str(e)

@app.post("/chat/")
async def chat(request_body: RequestBody):
    """Process user input and return the most relevant database result or generate a response from OpenAI."""
    query = request_body.query  # Extract query from the request body
    print(f"Received query: {query}")
    
    # Generate embedding for the query and reduce its dimension to 512
    query_embedding = generate_embedding(query)
    query_embedding = reduce_embedding_dimension(query_embedding)

    # Perform similarity search to find the closest matches in the database
    top_matches = similarity_search(query_embedding, top_n, MIN_SIMILARITY)

    if top_matches:
        # Use the content of the top matches as context for OpenAI's response
        context = " ".join([match[1] for match in top_matches])  # Combine the top contents into a single string
        print(f"Found context: {context[:30]}...")  # Display first 30 characters for brevity
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
