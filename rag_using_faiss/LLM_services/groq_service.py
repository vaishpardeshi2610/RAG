import os  # Add this import to access environment variables
from groq import Groq
from dotenv import load_dotenv
from models.pydantic_models import Metadata, RetrievedDocument, RetrievalResult
import json

load_dotenv()

# Initialize the GROQ client
groq_api_key = os.getenv("GROQ_API_KEY")  # Ensure this line retrieves the API key
groq_client = Groq(api_key=groq_api_key)  # Initialize the client with the API key

def generate_response(prompt):
    """Generate a chat response from the GROQ API based on the provided prompt."""
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="mixtral-8x7b-32768",  # Updated model name
        max_tokens=500,
        temperature=0.7,
    )

    # Extract the response content
    return chat_completion.choices[0].message.content  # Adjust based on the actual response structure

def query_and_generate_response(query, documents, faiss_index, model):
    """Query the document and generate a response using the GROQ API."""
    # Generate embedding for the query
    query_embedding = model.encode(query)

    # Perform a search in the FAISS index
    distances, indices = faiss_index.search(query_embedding)  # Use the FAISS index to search

    # Get the most similar document
    most_similar_index = indices[0][0]
    relevant_document = documents[most_similar_index]

    # Generate a response using the GROQ API
    prompt = f"Based on the following document, answer the question:\n\nDocument: {relevant_document}\n\nQuestion: {query}\n\nAnswer:"
    response = generate_response(prompt)  # Use the user query here

    # Create the retrieval result using Pydantic models
    retrieval_result = RetrievalResult(
        query=query,
        retrieved_documents=[
            RetrievedDocument(
                document_id=f"doc{most_similar_index + 1}",  # Assuming document IDs are sequential
                content=relevant_document,
                metadata=Metadata(
                    source="Attention is all you need",  # Replace with actual source if available
                    author="author_name"   # Replace with actual author if available
                )
            )
        ],
        retrieval_method="cosine similarity"  # Update retrieval method
    )
    
    # Use json.dumps for pretty printing
    print(json.dumps(retrieval_result.model_dump(), indent=2))  # Print the retrieval result in JSON format

    return response, distances[0][0]  # Return the response and similarity score