import random 

class Fake_llm:

    def __init__(self):
        print("Fake LLM initialized")

    def predict (self,prompt:str):

        response_list = [
            "The product is great and I am satisfied with it",
            "The product is not great and I am not satisfied with it",
            "The product is okay and I am neutral about it"
        ]
        return random.choice(response_list)