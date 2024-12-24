from fastapi import FastAPI
from pydantic import BaseModel
import openai
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Set OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Define the request model for the query
class QueryRequest(BaseModel):
    query: str
