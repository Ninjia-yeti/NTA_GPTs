import openai
import psycopg2

openai.api_key = 'your_openai_api_key'

def generate_embedding(text):
    """Generate embeddings for the given text using OpenAI."""
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response['data'][0]['embedding']

def insert_into_db(content, file_name, chunk_index, embedding):
    """Insert vector data into PostgreSQL."""
    conn = psycopg2.connect(
        host="localhost",
        database="nta_data",
        user="postgres",
        password="your_postgres_password"
    )
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO vectors (embedding, content, file_name, chunk_index)
        VALUES (%s, %s, %s, %s)
    """, (embedding, content, file_name, chunk_index))
    conn.commit()
    cursor.close()
    conn.close()

# Load and split your text data
with open("nta_data.txt", "r", encoding="utf-8") as f:
    large_text = f.read()

# Split text into chunks (e.g., by paragraphs)
chunks = large_text.split("\n\n")  # Adjust splitting as needed

for i, chunk in enumerate(chunks):
    embedding = generate_embedding(chunk)
    insert_into_db(chunk, "nta_data.txt", i, embedding)
