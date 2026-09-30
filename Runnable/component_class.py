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


class fake_promptTemplate:
    def __init__(self, template:str, input_variables):
        self.template = template
        self.input_variables = input_variables

    def format(self , input_dict):
        return self.template.format(**input_dict)
