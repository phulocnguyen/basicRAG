from typing import Union
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class VectorDB:
    def __init__(self, documents = None, vector_db: Union[Chroma, FAISS] = Chroma, 
                 embeddings = HuggingFaceEmbeddings()) -> None:
        self.vector_db = vector_db
        self.embeddings = HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")

        self.db = self.build_db(documents)
    def build_db(self, documents):
        db = self.vector_db.from_documents(documents=documents, embedding = self.embeddings)
        return db
    
    def get_retriever(self, search_type="similarity", search_kwargs: dict = {"k":3}):
        retriever = self.db.as_retriever(search_type=search_type ,search_kwargs=search_kwargs)
        return retriever
