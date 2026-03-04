import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

class GeminiManager:
    """Class responsible for initializing Google Gemini models."""
    
    def __init__(self, embedding_model_name: str = None, chat_model_name: str = "gemini-2.5-flash-lite"):
        self.embedding_model_name = embedding_model_name or os.getenv("GOOGLE_EMBEDDING_MODEL")
        self.chat_model_name = chat_model_name or os.getenv("GOOGLE_CHAT_MODEL")
        
        if not self.embedding_model_name:
            raise ValueError("The embedding model name must be provided.")

    def get_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        print(f"Initializing Gemini Embeddings model: {self.embedding_model_name}")
        return GoogleGenerativeAIEmbeddings(model=self.embedding_model_name)
    
    def get_chat_model(self) -> ChatGoogleGenerativeAI:
        print(f"Initializing Gemini Chat model: {self.chat_model_name}")
        return ChatGoogleGenerativeAI(model=self.chat_model_name, temperature=0.0)
        