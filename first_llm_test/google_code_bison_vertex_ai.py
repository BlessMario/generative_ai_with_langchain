import sys
from pathlib import Path
from langchain_community.llms import VertexAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

from langchain_core.prompts import PromptTemplate


question = """
Given an integer n, return a string array answer (1-indexed) where:

answer[i] == "FizzBuzz" if i is divisible by 3 and 5.
answer[i] == "Fizz" if i is divisible by 3.
answer[i] == "Buzz" if i is divisible by 5.
answer[i] == i (as a string) if none of above conditions are true.
"""

prompt_template = "Code me a {question} in Python"
prompt = PromptTemplate(
    input_variables=["question"], template=prompt_template
)

llm = VertexAI(model_name="code-bison")
llm_chain = LLMChain(prompt=prompt, llm=llm)
print(llm_chain.run(question))