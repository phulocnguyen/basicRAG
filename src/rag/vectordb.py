from typing import Union
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

class VectorDB:
    def __init__(self, documents = None, vector_db: Union[Chroma, FAISS] = Chroma, 
                 embeddings = HuggingFaceEmbeddings()) -> None:
        self.vector_db = vector_db
        self.embeddings = embeddings
        self.db = self.vector_db.from_documents(documents=documents, embedding = self.embeddings)
    
    def retrieval(self, search_type="similarity", search_kwargs: dict = {"k":10}):
        retriever = self.db.as_retirever(search_type=search_type ,search_kwargs=search_kwargs)
        return retriever
