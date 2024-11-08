from pydantic import BaseModel
from typing import List

class Metadata(BaseModel):
    source: str
    author: str

class RetrievedDocument(BaseModel):
    document_id: str
    content: str
    metadata: Metadata

class RetrievalResult(BaseModel):
    query: str
    retrieved_documents: List[RetrievedDocument]
    retrieval_method: str
