import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-3.5-turbo")
CHROMA_DB_URL = os.getenv("CHROMA_DB_URL", "http://localhost:8000")
RERANK_MODEL = os.getenv("RERANK_MODEL", "rerank-model")
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL')
PERSIST_DIRECTORY = os.getenv('PERSIST_DIRECTORY')