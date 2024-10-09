from langchain.llms.huggingface_pipeline import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_compute_dtype=torch.bfloat16)

def get_LLM(model_name: str = "google/gemma-2-2b-jpn-it",
            max_new_tokens=1024,
            **kwargs):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=nf4_config,
        low_cpu_mem_usage=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id
    )
    llm = HuggingFacePipeline(pipeline=model_pipeline, model_kwargs=kwargs)
    return llm
