import os

from dotenv import load_dotenv

class ConfigValidator:
    """Class responsible for loading and validating environment variables."""
    
    REQUIRED_KEYS = (
        "GOOGLE_API_KEY", 
        "GOOGLE_EMBEDDING_MODEL", 
        "DATABASE_URL",
        "PG_VECTOR_COLLECTION_NAME", 
        "PDF_PATH"
    )

    @classmethod
    def load_and_validate(cls) -> dict:
        print("Loading environment variables from .env file...")
        load_dotenv()
        
        config = {}
        for key in cls.REQUIRED_KEYS:
            value = os.getenv(key)
            if not value:
                raise RuntimeError(f"Environment variable {key} is not set.")
            config[key] = value

        cls._check_pdf_path(config["PDF_PATH"])
        return config

    @staticmethod
    def _check_pdf_path(pdf_path: str):
        print(f"Checking PDF path: {pdf_path}")
        if not os.path.exists(pdf_path):
            raise RuntimeError(f"The PDF file {pdf_path} does not exist.")