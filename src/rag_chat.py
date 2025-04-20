from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
from config import BASE_MODEL, MY_MODEL, HF_TOKEN
import numpy as np

class SchoolChatbot:
    def __init__(self):
        model_id = MY_MODEL if MY_MODEL else BASE_MODEL
        self.client = InferenceClient(model=model_id, token=HF_TOKEN)

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index("faiss_index-1.index")
        self.contexts = pd.read_csv("contexts.csv")["context"].tolist()

    def retrieve_context(self, user_input, k=3):
        query_embedding = self.embedder.encode([user_input])
        _, indices = self.index.search(query_embedding, k)
        return "\n".join([self.contexts[i] for i in indices[0]])

    def format_prompt(self, user_input):
        print(f'user input {user_input} {type(user_input)}')
        context = self.retrieve_context(user_input)
        return (
            "You are a helpful assistant that specializes in Boston Public Schools.\n\n"
            f"Context:\n{context}\n\n"
            f"User: {user_input}\n"
            "Assistant:"
        )

    def get_response(self, user_input):
        prompt = self.format_prompt(user_input)
        response = self.client.text_generation(prompt, max_new_tokens=300, temperature=0.5)
        return response.strip()