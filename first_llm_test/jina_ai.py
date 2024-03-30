import sys
from langchain.chat_models import JinaChat
from langchain.schema import HumanMessage

from pathlib import Path
# Add the root folder to sys.path
root_path = Path(__file__).resolve().parents[1]  # Adjust the number of parents based on your structure
sys.path.append(str(root_path))
from config import set_environment

set_environment()

chat = JinaChat(temperature=0.)

message=[HumanMessage(
    content="Translate this sentence from English to French: I love generative AI!"
    )

]

print(chat(message))


from langchain.schema import SystemMessage
chat = JinaChat(temperature=0.)

print(chat(
    [
        SystemMessage(
            content="You help a user find a nutritious and tasty food to eat in one word."            
        ),HumanMessage(
            content="I' like pastea with cheese, but I need to eat more vegetables,  What shoud I eat?"
        )   
    ]
))