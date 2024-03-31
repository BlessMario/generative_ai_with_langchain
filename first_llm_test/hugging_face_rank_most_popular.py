from huggingface_hub import list_models

def list_most_popular(task: str):
    for rank, model in enumerate(list_models(filter=task, sort='downloads', direction=-1, limit=10)):
        if rank == 5:
            break;
        print(f"{model.id}, {model.downloads}")
        #print(f"{rank + 1}. {model}")


list_most_popular("text-classification")