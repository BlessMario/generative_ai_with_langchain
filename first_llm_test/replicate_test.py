import sys
from langchain.llms import Replicate
from pathlib import Path
# Add the root folder to sys.path
root_path = Path(__file__).resolve().parents[1]  # Adjust the number of parents based on your structure
sys.path.append(str(root_path))
from config import set_environment

set_environment()

text2image = Replicate( model="stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
    input={
        "width": 768,
        "height": 768,
        "prompt": "an astronaut riding a horse on mars, hd, dramatic lighting",
        "scheduler": "K_EULER",
        "num_outputs": 1,
        "guidance_scale": 7.5,
        "num_inference_steps": 50
    }
)
print(text2image("an astronaut riding a horse on mars, hd, dramatic lighting"))

text2image_2 = Replicate(model="stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf", input={'image_dimensions': '512x512'})
print(text2image_2("a book cover for a book about creating generative ai applications in Python"))
image_url = text2image_2("a book cover for a book about creating generative ai applications in Python"
)

print(image_url)