import os
import json
from groq import Groq
from pydantic_models.retrieval import retrieve_relevant_documents
from pydantic_models.models import Document, RetrievalResult

# Initialize the Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_response_with_groq(conn, query, model="llama3-8b-8192"):
    """
    Generates a response using Groq's chat completion model based on the retrieved context.
    """
    relevant_docs = retrieve_relevant_documents(conn, query)
    context = "\n".join([doc.content for doc in relevant_docs.retrieved_documents])  # Combine retrieved documents into context
    
    # Prepare the messages for the Groq API
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        }],
        model=model,
    )
    
    # Prepare the output structure
    output = {
        "retrieval_result": {
            "query": query,
            "retrieved_documents": [
                {
                    "document_id": doc.document_id,
                    "content": doc.content,
                    "metadata": {
                        "source": doc.metadata.source,
                        "author": doc.metadata.author
                    }
                } for doc in relevant_docs.retrieved_documents
            ],
            "retrieval_method": relevant_docs.retrieval_method
        }
    }

    # Print the output in the specified format
    print(json.dumps(output, indent=2))  # Pretty print the JSON output

    # Return the generated response
    return chat_completion.choices[0].message.content
