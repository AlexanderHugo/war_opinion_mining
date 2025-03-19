import os
from huggingface_hub import InferenceClient

class FalconLLM:
    __token = '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'
    def __init__(self, model = "tiiuae/falcon-7b-instruct"):
        client = InferenceClient(
            model = model,
            token = self.__token
        )

    def text_generation(self, question):
        return self.client.text_generation(question)

