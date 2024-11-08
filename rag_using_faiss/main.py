from sentence_transformers import SentenceTransformer
from data.document_loader import extract_text_from_pdf
import numpy as np
from db_config.db_services import DatabaseConfig  # Import the DatabaseConfig class
from LLM_services.groq_service import query_and_generate_response  # Import the new function
from vector_store.faiss_index import FAISSIndex  # Import the FAISSIndex class
# Load your pre-trained embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Use the extract_text_from_pdf function to load the document text
pdf_path = "attention_is.pdf"  # Replace with your PDF file path
documents = extract_text_from_pdf(pdf_path)

# Generate embeddings
embeddings = model.encode(documents)

# Create a FAISS index for efficient similarity search
faiss_index = FAISSIndex(embeddings)  # Create an instance of FAISSIndex

# Database configuration
db_config = DatabaseConfig()  # No parameters needed, reads from .env

# Connect to the database and create the table
db_config.connect()
db_config.create_table()

# Insert embeddings into PostgreSQL
db_config.insert_embeddings(documents, [embedding.tolist() for embedding in embeddings])  # Convert embeddings to list for PostgreSQL

# Example usage of the query_and_generate_response function
if __name__ == "__main__":
    user_query = "Why are transformers important?"  # Prompt the user for input
    response_text, similarity_score = query_and_generate_response(user_query, documents, faiss_index, model)  # Pass necessary arguments

    # Display the response
    print("----------------------------------------------------------\n")
    print(f"Embeddings:\n{embeddings}")
    print("----------------------------------------------------------\n")
    print(f"Response:\n{response_text}\n")
    print("----------------------------------------------------------\n")
    print(f"Similarity Score: {similarity_score}")
    
    # Close the database connection
    db_config.close()