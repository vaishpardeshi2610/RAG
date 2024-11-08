import faiss
import numpy as np

class FAISSIndex:
    def __init__(self, embeddings):
        # Normalize the embeddings to unit length for cosine similarity
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embedding_dimension = normalized_embeddings.shape[1]  # Get the dimension of the embeddings
        self.index = faiss.IndexFlatIP(self.embedding_dimension)  # Create a FAISS index for inner product
        self.index.add(normalized_embeddings)  # Add normalized embeddings to the index

    def search(self, query_embedding, k=1):
        """Search for the nearest neighbors in the FAISS index using cosine similarity."""
        # Normalize the query embedding
        normalized_query = query_embedding / np.linalg.norm(query_embedding)
        distances, indices = self.index.search(np.array([normalized_query]), k)  # Search for the nearest neighbor
        return distances, indices
