import sys
from pathlib import Path
from langchain_community.llms import OpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import HuggingFaceHub
from langchain_community.utilities import PythonREPL
from langchain_experimental.tools import PythonREPLTool


# Add the root folder to sys.path
root_path = Path(__file__).resolve().parents[1]  # Adjust the number of parents based on your structure
sys.path.append(str(root_path))

from config import set_environment
set_environment()

llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 64})
prompt = "In whick country is Arusha?"  
completion = llm(prompt)
print(completion)