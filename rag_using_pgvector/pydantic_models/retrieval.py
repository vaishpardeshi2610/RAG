from typing import List
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import psycopg2

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

class DocumentMetadata(BaseModel):
    source: str
    author: str

class Document(BaseModel):
    document_id: str
    content: str
    metadata: DocumentMetadata

class RetrievalResult(BaseModel):
    query: str
    query_embedding: List[float]
    retrieved_documents: List[Document]
    retrieval_method: str

def retrieve_relevant_documents(conn, query, top_k=3, similarity_threshold=0.75):
    query_embedding = model.encode(query).tolist()
    cur = conn.cursor()

    # Execute SQL to retrieve documents based on similarity
    cur.execute(""" 
    SELECT document, 1 - (embedding <=> %s::vector) AS similarity 
    FROM knowledge_base_embeddings 
    ORDER BY similarity DESC 
    LIMIT %s; 
    """, (query_embedding, top_k))
    
    results = cur.fetchall()

    # Prepare the retrieval result
    retrieved_documents = []
    seen_documents = set()

    for idx, result in enumerate(results):
        document, similarity = result
        # Only include documents that have a similarity above the threshold and aren't duplicates
        if similarity >= similarity_threshold and document not in seen_documents:
            seen_documents.add(document)
            # Create a document entry with metadata
            doc_entry = Document(
                document_id=f"doc{idx + 1}",  # Assigning a document ID
                content=document,
                metadata=DocumentMetadata(source="knowledge_base", author="Unknown")  # Placeholder metadata
            )
            retrieved_documents.append(doc_entry)

    # Create the retrieval result
    retrieval_result = RetrievalResult(
        query=query,
        query_embedding=query_embedding,
        retrieved_documents=retrieved_documents,
        retrieval_method="vector search"
    )

    # Return the retrieval result
    return retrieval_result
