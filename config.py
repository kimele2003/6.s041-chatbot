import os
from dotenv import load_dotenv

# Load from .env file. Store your HF token in the .env file.
load_dotenv()

BASE_MODEL = "google/flan-t5-large" # fast and good for Q&A
# Other options:
# BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"
# BASE_MODEL = "openlm-research/open_llama_3b"

# If you finetune the model or change it in any way, save it to huggingface hub, then set MY_MODEL to your model ID. The model ID is in the format "your-username/your-model-name".
MY_MODEL = None

HF_TOKEN = os.getenv("HF_TOKEN")
