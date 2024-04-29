
import sys
import os
from pathlib import Path
import streamlit as st

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

print(torch.backends.cudnn.enabled)
print(torch.cuda.is_available())
# Add the root folder to sys.path
root_path = Path(__file__).resolve().parents[1]  # Adjust the number of parents based on your structure
sys.path.append(str(root_path))
from config import set_environment
set_environment()

ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

print(torch.cuda.is_available())
model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=ACCESS_TOKEN)
quantization_config = BitsAndBytesConfig(load_in_4bit=True, 
                                         bnb_4bit_compute_dtype=torch.bfloat16)

model = AutoModelForCausalLM.from_pretrained(model_id, 
                                             device_map="auto", 
                                             quantization_config=quantization_config,
                                             token=ACCESS_TOKEN)
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'




def generate(question: str, context: str):
    if context == None or context == "":
        prompt = f"""Give a detailed answer to the following question. Question: {question}"""
    else:
        prompt = f"""Using the information contained in the context, give a detailed answer to the question.
            Context: {context}.
            Question: {question}"""
    chat = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer.encode(
        formatted_prompt, add_special_tokens=False, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=250,
            do_sample=False,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    response = response[len(formatted_prompt) :]  # remove input prompt from reponse
    response = response.replace("<eos>", "")  # remove eos token
    return response


 print(generate(question="How are you?", context=""))


