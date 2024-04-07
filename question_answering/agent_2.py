
import sys
from pathlib import Path
import streamlit as st
from langchain_community.callbacks import StreamlitCallbackHandler
from typing import Literal

from langchain.agents import initialize_agent, load_tools, AgentType
from langchain.chains.base import Chain
from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.plan_and_execute import (
    load_chat_planner, load_agent_executor, PlanAndExecute
)

from question_answering.agent import load_agent
from question_answering.utils import MEMORY

# Add the root folder to sys.path
root_path = Path(__file__).resolve().parents[1]  # Adjust the number of parents based on your structure
sys.path.append(str(root_path))
from config import set_environment

set_environment()

st.set_page_config(page_title="LangChain Question Answering", page_icon=":robot:")
st.header("Ask a research question!")

ReasoningStrategies = Literal["zero-shot-react", "plan-and-solve"]

strategy = st.radio(
    "Reasoning strategy",
    ("plan-and-solve", "zero-shot-react", ))

tool_names = st.multiselect(
    'Which tools do you want to use?',
    [
        "google-search", "ddg-search", "wolfram-alpha", "arxiv",
        "wikipedia", "python_repl", "pal-math",
        "llm-math"
    ],
    ["ddg-search", "wolfram-alpha", "wikipedia"])
if st.sidebar.button("Clear message history"):
    MEMORY.chat_memory.clear()

avatars = {"human": "user", "ai": "assistant"}
for msg in MEMORY.chat_memory.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

assert strategy is not None
agent_chain = load_agent(tool_names=tool_names, strategy=strategy)

assistant = st.chat_message("assistant")
if prompt := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(prompt)
    stream_handler = StreamlitCallbackHandler(assistant)
    with st.chat_message("assistant"):
        response = agent_chain.run({
            "input": prompt,
            "chat_history": MEMORY.chat_memory.messages
        }, callbacks=[stream_handler]
        )

def load_agent(
        tool_names: list[str],
        strategy: ReasoningStrategies = "zero-shot-react"
) -> Chain:
    llm = ChatOpenAI(temperature=0, streaming=True)
    tools = load_tools(
        tool_names=tool_names,
        llm=llm
    )
    if strategy == "plan-and-solve":
        planner = load_chat_planner(llm)
        executor = load_agent_executor(llm, tools, verbose=True)
        return PlanAndExecute(planner=planner, executor=executor, verbose=True)

    return initialize_agent(
        tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
