from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from base.llm_model import get_hf_llm
from rag.main import build_rag_chain, InputQA, OutputQA

llm = get_hf_llm(temperature=0.9)
genai_docs = "/Users/phulocnguyen/Documents/Workspace/basicRAG/datasource/gen_ai"

# -------- Chains --------
genai_chain = build_rag_chain(llm, data_dir=genai_docs, data_type="pdf")

# -------- App - FastAPI --------
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# -------- Routes - FastAPI --------
@app.get("/check")
async def check():
    return {"status": "ok"}

@app.post("/gen_ai", response_model=OutputQA)
async def generative_ai(inputs: InputQA):
    answer = await genai_chain.ainvoke(inputs.question)
    return {"answer": answer}

# -------- Langserve Routes - Playground --------
add_routes(
    app,
    genai_chain,
    playground_type="default",
    path="/gen_ai",
)