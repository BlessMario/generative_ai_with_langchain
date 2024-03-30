import sys
from transformers import pipeline
import torch

generate_text = pipeline(
    model= "aisquared/dlite-v1-355m",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    framwork="pt"
)

# generate_text("In this chapter, we'll discuss first steps with generative AI in Python.")

from pathlib import Path
# Add the root folder to sys.path
root_path = Path(__file__).resolve().parents[1]  # Adjust the number of parents based on your structure
sys.path.append(str(root_path))
from config import set_environment

set_environment()

#from langchain import PromptTemplate, LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm = LLMChain(prompt=prompt, llm=generate_text)
question = "What is electroencephalography?"
