from langchain_community.llms import FakeListLLM
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType, Tool
from langchain_community.utilities import PythonREPL
from langchain_experimental.tools import PythonREPLTool
from langchain_community.llms import FakeListLLM
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType, Tool
from langchain_community.utilities import PythonREPL
from langchain_experimental.tools import PythonREPLTool

responses = ["Action: Python_REPL\nAction Input: print(2 + 2)\nAction Output: 4"]

agent = initialize_agent(
    tools=[PythonREPLTool()],
    llm=FakeListLLM(responses=responses),
)

agent.invoke("2 + 2")
agent.invoke("what is 2 + 2?")


# agent = initialize_agent(
#     tools=[python_repl],
#     llm=llm,
# )


# agent.invoke("what is 2 + 2?")