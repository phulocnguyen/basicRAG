from rag.main import build_rag_chain
from base.llm_model import get_LLM

def run():
    llm_model = get_LLM(temperature=0.8)
    docs = "./datasource/gen_ai"
    chain = build_rag_chain(llm_model, docs, "pdf")

    query = "How attention works?"  
    output = chain.invoke(query)
    print(output)

run()


