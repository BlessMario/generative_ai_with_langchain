import sys
from pathlib import Path
from langchain_community.llms import VertexAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

from langchain_core.prompts import PromptTemplate

# from langchain.agents import load_tools
# from langchain.agents import initialize_agent
# from langchain.agents import AgentType
# from langchain.llms import HuggingFaceHub
# from langchain_community.utilities import PythonREPL
# from langchain_experimental.tools import PythonREPLTool


template = """ Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])  

llm = VertexAI()
llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

question = "What NFL team won the Supoer Bowl in the year Justin Beiber was born?"

print(llm_chain.run(question))

