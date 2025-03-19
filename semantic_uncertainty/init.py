"""
Initialization module for Hugging Face authentication.
"""
from huggingface_hub import login
import os

def initialize_huggingface() -> None:
    """
    Initialize Hugging Face authentication using the stored token.
    This function should be called before any Hugging Face operations.
    """
    # Store your token here
    HUGGING_FACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')  # Replace with your token

    # Initialize Hugging Face authentication
    login(token=HUGGING_FACE_TOKEN)

