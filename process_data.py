import os
import openai
import psycopg2
import numpy as np
from dotenv import load_dotenv
from tiktoken import encoding_for_model

# Load environment variables from .env
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load the tokenizer for the specific model
tokenizer = encoding_for_model("text-embedding-ada-002")
MAX_TOKENS = 500  # Token limit for the model

def tokenize_and_chunk(text, max_tokens):
    """Split text into chunks based on token limits."""
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk) for chunk in chunks]

def generate_embedding(text):
    """Generate embeddings for the given text using OpenAI."""
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    embedding = response['data'][0]['embedding']
    # Convert the embedding to a numpy array with the shape (1, 1536)
    embedding = np.array(embedding).reshape(1, -1)
    return embedding

def insert_into_db(content, file_name, chunk_index, embedding):
    """Insert vector data into PostgreSQL."""
    # Flatten the embedding before storing (pgvector requires a 1D array)
    embedding_flat = embedding.flatten()
    conn = psycopg2.connect(
        host="localhost",
        database="nta_data",
        user="postgres",
        password="nta_gpts"
    )
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO vectors (embedding, content, file_name, chunk_index)
        VALUES (%s, %s, %s, %s)
    """, (embedding_flat.tolist(), content, file_name, chunk_index))  # Convert numpy array to list
    conn.commit()
    cursor.close()
    conn.close()

def process_text_files(folder_path):
    """Process all text files in the folder and store embeddings in the database."""
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        if not os.path.isfile(file_path) or not file_name.endswith(".txt"):
            continue  # Skip non-text files or directories
        
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Split the content into tokenized chunks
        chunks = tokenize_and_chunk(content, MAX_TOKENS)
        
        for chunk_index, chunk in enumerate(chunks):
            try:
                # Generate embedding and insert into the database
                embedding = generate_embedding(chunk)
                insert_into_db(chunk, file_name, chunk_index, embedding)
                print(f"Inserted chunk {chunk_index} of {file_name} into the database.")
            except Exception as e:
                print(f"Error processing chunk {chunk_index} of {file_name}: {e}")

# Main execution
if __name__ == "__main__":
    folder_path = "./NTA_database_test"  # Folder containing text files
    process_text_files(folder_path)
