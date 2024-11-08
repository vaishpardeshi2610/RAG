from pydantic import BaseModel
from typing import List

class DocumentMetadata(BaseModel):
    source: str
    author: str

class Document(BaseModel):
    document_id: str
    content: str
    metadata: DocumentMetadata

class RetrievalResult(BaseModel):
    query: str
    retrieved_documents: List[Document]
    retrieval_method: str
