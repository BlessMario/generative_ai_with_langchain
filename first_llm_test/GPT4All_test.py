from langchain_community.llms import GPT4All
model = GPT4All(model="/home/mario/Downloads/mistral-7b-openorca.gguf2.Q4_0.gguf", device='cpu', n_threads=8, verbose=True)
response = model(
    "We can run large language models locally for all kinds of applications, "
)
print(response)

output = model.generate(prompts=["The capital of France is "], max_tokens=3)
print(output)

