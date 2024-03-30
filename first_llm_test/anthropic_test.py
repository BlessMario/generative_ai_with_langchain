import sys
import os
import anthropic

from pathlib import Path
# Add the root folder to sys.path
root_path = Path(__file__).resolve().parents[1]  # Adjust the number of parents based on your structure
sys.path.append(str(root_path))
from config import set_environment

set_environment()

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)
message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=4000,
    temperature=0.7,
    system="Ich bin ein Drehbuchautor und möchte eine kurz Geschichte über 2 Jungs verfilmen, mit kurzen Szenen, Film sollte 5 min gehen.",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Der Film sollte als Inhalt ein Schatz enthalten den die 2 Jungs (6 und 11 Jahre Alt) im Wald finden. Sie suchen den Besitzer finden, jedoch finden sie heraus dass der Schatz auf späte Mittelalter zurück geht."
                }
            ]
        }
    ]
)
print(message.content)