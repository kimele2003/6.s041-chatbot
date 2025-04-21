import torch
from huggingface_hub import login

from src.rag_chat import SchoolChatbot
from config import BASE_MODEL, MY_MODEL

login()

chatbot = SchoolChatbot()

test_question = "I live in Jamaica Plain and want to send my child to a school that offers Spanish classes. What schools are available?"

print(f"\nQuestion: {test_question}")
response = chatbot.get_response(test_question)
print(f"Response: {response}")

test_question = "Could you tell me more about the Hernandez School? I've heard good things about it."

print(f"\nQuestion: {test_question}")
response = chatbot.get_response(test_question)
print(f"Response: {response}")