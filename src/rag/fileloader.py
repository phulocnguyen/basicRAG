from typing import Union, List, Literal
import glob
import os
from tqdm import tqdm
import multiprocessing
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def remove_non_utf8_characters(text):
    return ' '.join(char for char in text if ord(char) < 128)

def load_pdf(pdf_file):
    docs = PyPDFLoader(pdf_file, extract_images=True).load()
    for doc in docs:
        doc.page_content = remove_non_utf8_characters(doc.page_content)
    return docs

def load_txt(txt_file):
    loader = TextLoader(txt_file, encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.page_content = remove_non_utf8_characters(doc.page_content)
    return docs

def get_num_cpu():
    return multiprocessing.cpu_count()

class BaseLoader:
    def __init__(self) -> None:
        self.num_processes = get_num_cpu()

    def __call__(self, files: List[str], **kwargs):
        pass

class PDFLoader(BaseLoader):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pdf_files: List[str], **kwargs):
        num_processes = min(self.num_processes, kwargs["workers"])
        with multiprocessing.Pool(processes=num_processes) as pool:
            doc_loaded = []
            with tqdm(total=len(pdf_files), desc="Loading PDFs", unit="file") as pbar:
                for result in pool.imap_unordered(load_pdf, pdf_files):
                    doc_loaded.extend(result)
                    pbar.update(1)
        return doc_loaded

class TXTLoader(BaseLoader):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, txt_files: List[str], **kwargs):
        num_processes = min(self.num_processes, kwargs["workers"])
        with multiprocessing.Pool(processes=num_processes) as pool:
            doc_loaded = []
            with tqdm(total=len(txt_files), desc="Loading TXTs", unit="file") as pbar:
                for result in pool.imap_unordered(load_txt, txt_files):
                    doc_loaded.extend(result)
                    pbar.update(1)
        return doc_loaded

class TextSplitter:
    def __init__(self,
                separators: List[str] = ['\n\n', '\n', ' ', ''],
                chunk_size: int = 300,
                chunk_overlap: int = 0) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def __call__(self, documents: List[Document]):
        return self.splitter.split_documents(documents)

class Loader:
    def __init__(self,
                file_type: Union[str, Literal["pdf", "txt"]] = "pdf",
                split_kwargs: dict = {
                    "chunk_size": 300,
                    "chunk_overlap": 0}) -> None:
        assert file_type in ["pdf", "txt"], "file_type must be 'pdf' or 'txt'"
        self.file_type = file_type
        if file_type == "pdf":
            self.doc_loader = PDFLoader()
        elif file_type == "txt":
            self.doc_loader = TXTLoader()
        self.doc_spltter = TextSplitter(**split_kwargs)

    def load(self, files: Union[str, List[str]], workers: int = 1):
        if isinstance(files, str):
            files = [files]
        doc_loaded = self.doc_loader(files, workers=workers)
        doc_split = self.doc_spltter(doc_loaded)
        return doc_split

    def load_dir(self, dir_path: str, workers: int = 1):
        if self.file_type == "pdf":
            files = glob.glob(os.path.join(dir_path, "*.pdf"))
        elif self.file_type == "txt":
            files = glob.glob(os.path.join(dir_path, "*.txt"))
        else:
            raise ValueError("Unsupported file_type")
        assert len(files) > 0, f"No {self.file_type} files found in {dir_path}"
        return self.load(files, workers=workers)
