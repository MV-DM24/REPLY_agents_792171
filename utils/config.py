# utils/config.py
import os
import sys
from dotenv import load_dotenv


load_dotenv()

class Config:
    """Configuration settings for the CrewAI project."""
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    MEMO_API_KEY = os.getenv("MEMO_API_KEY")
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "my_chroma_db")  # Provide a default
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "my_crew_collection")  # Provide a default

    AVAILABLE_DATA_PATHS = {
        'AMMINISTRATI.csv': os.getenv("AMMINISTRATI"),
        'REDDITO.csv': os.getenv("REDDITO"),
        'PENDOLARISMO.csv': os.getenv("PENDOLARISMO"),
        'STIPENDI.csv': os.getenv("STIPENDI")
    }

    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")  # default model


    def validate_config(self):
        """Validates that essential configuration variables are set."""
        if not self.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set in the environment.")
        if not all(self.AVAILABLE_DATA_PATHS.values()):
            raise ValueError("All data paths in AVAILABLE_DATA_PATHS must be set in the environment.")

config = Config()
try:
    config.validate_config()
    print("Configuration loaded and validated successfully.")
except ValueError as e:
    print(f"Configuration error: {e}")
    # Handle the error appropriately - e.g., exit the program, use default values, etc.
    # For now, we'll just re-raise the exception to halt execution
    raise