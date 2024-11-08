import os
import psycopg2
from dotenv import load_dotenv
from dg_config.database import ensure_pgvector_extension, create_embeddings_table, close_connection
from embeddings.embeddings import store_embeddings
from pydantic_models.retrieval import retrieve_relevant_documents
from LLM_services.response import generate_response_with_groq

# Load environment variables from .env file
load_dotenv()

# Database and API configurations
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Connect to PostgreSQL database
conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST)

# Ensure pgvector extension exists
ensure_pgvector_extension(conn)

# Ensure the table for embeddings exists
create_embeddings_table(conn)

# Store embeddings if not already stored
store_embeddings(conn)

# Sample query
query = "What is a function in programming?"
relevant_docs = retrieve_relevant_documents(conn, query)  # Get relevant documents

# Display the retrieved embeddings
print("\nRetrieved Embeddings:")
for doc in relevant_docs.retrieved_documents:
    print(f"Document ID: {doc.document_id}, Content: {doc.content}, Metadata: {doc.metadata}")

# Generate response
answer = generate_response_with_groq(conn, query)

print("Generated Answer:", answer)

# Close the database connection
close_connection(conn)
