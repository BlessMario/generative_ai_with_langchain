import sys
from pathlib import Path
from langchain_community.llms import OpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain_community.utilities import PythonREPL
from langchain_experimental.tools import PythonREPLTool


# Add the root folder to sys.path
root_path = Path(__file__).resolve().parents[1]  # Adjust the number of parents based on your structure
sys.path.append(str(root_path))

from config import set_environment

#from ..config import set_environment

set_environment()

llm = OpenAI(temperature=0., model ="babbage-002")

#tools = load_tools(['python_execution'])
tools=[PythonREPLTool()]

agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True, verbose=True)

agent.run("what's the capital of Tanzania?")