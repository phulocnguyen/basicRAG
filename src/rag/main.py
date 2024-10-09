from rag.dataloader import Loader
from rag.vectordb import VectorDB 
from rag.offlinerag import offlinerag

def build_rag_chain(llm, data_dir, data_type):
    doc_loaded = Loader(file_type=data_type).load_dir(data_dir, workers=2) 
    retriever = VectorDB(documents = doc_loaded).retrieval()
    rag_chain = offlinerag(llm).make_chain(retriever)
    return rag_chain


