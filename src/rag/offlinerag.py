from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import re
from langchain_core.prompts import PromptTemplate


class StrOutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()

    def parse(self, text: str) -> str:
        return self.extract_answer(text)

    def extract_answer(self, text_response: str, pattern: str = r"Answer:\s*(.*)") -> str:
        match = re.search(pattern, text_response, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            return answer_text
        else:
            return text_response

class Offline_RAG:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.str_parser = StrOutputParser()

    def get_chain(self, retriever):
        prompt = PromptTemplate.from_template(
        "Use the following context to answer the question concisely:\n\n{context}\n\nQuestion: {question}\nAnswer:"
    )
        input_data = {
        "context": retriever | self.format_docs,  
        "question": lambda x: x  
        }
        return input_data | prompt | self.llm | self.str_parser

    def format_docs(self, docs):
        context = "\n\n".join(doc.page_content for doc in docs)
        print("=== Retrieved context ===")
        print(context)
        print("=========================")
        return context
