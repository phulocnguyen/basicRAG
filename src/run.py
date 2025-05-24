import sys
import multiprocessing
from base.llm_model import get_hf_llm
from rag.main import build_rag_chain

def main(input_path, output_path):
    temperature = 0.5
    data_dir = "data/datasource"

    multiprocessing.set_start_method("spawn", force=True)

    print("Loading LLM...")
    llm = get_hf_llm(temperature=temperature)
    print(f"Building RAG chain with docs in '{data_dir}'...")
    rag_chain = build_rag_chain(llm, data_dir=data_dir, data_type="txt")
    print("Setup complete. Processing input file...")

    with open(input_path, "r", encoding="utf-8") as fin:
        questions = [line.strip() for line in fin if line.strip()]

    answers = []
    for question in questions:
        print(f"Question: {question}")
        try:
            result = rag_chain.invoke(question)
            answer = str(result)
            print(f"Answer: {answer}")
            answers.append(answer.replace("\n", " "))  
        except Exception as e:
            print(f"Error during inference: {e}", file=sys.stderr)
            answers.append(f"Error: {e}")

    with open(output_path, "w", encoding="utf-8") as fout:
        for ans in answers:
            fout.write(ans + "\n")

    print(f"Finished! Answers saved to '{output_path}'")

if __name__ == "__main__":

    input_file = "data/test/questions.txt"
    output_file = "data/test/system_output.txt"

    main(input_file, output_file)
