from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
def get_hf_llm(model_name: str = "Qwen/Qwen3-0.6B", max_new_token=128, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(model_name)

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_token,
        pad_token_id=tokenizer.eos_token_id,
        device_map='auto'
    )

    llm = HuggingFacePipeline(pipeline=model_pipeline, model_kwargs=kwargs)
    return llm

