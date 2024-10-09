import re
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class Str_OutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()
    def parse(self, text: str) -> str:
        return self.extract_answer(text)
    def extract_answer(self, response:str,
                       pattern: str = r"Answer:\s*(.*)") -> str:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            return answer
        else: 
            return response

class offlinerag:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = hub.pull("rlm/rag-prompt")
        self.str_parser = Str_OutputParser
    
    def make_chain(self, retriever):
        input_data = {
            "context": retriever | self.format_docs,
            "question": RunnablePassthrough()
        }
        rag_chain = (
            input_data | self.prompt | self.llm | self.str_parser
        )

        return rag_chain
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)