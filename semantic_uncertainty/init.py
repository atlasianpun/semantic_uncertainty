"""
Initialization module for Hugging Face authentication.
"""
from huggingface_hub import login

def initialize_huggingface() -> None:
    """
    Initialize Hugging Face authentication using the stored token.
    This function should be called before any Hugging Face operations.
    """
    # Store your token here
    HUGGING_FACE_TOKEN = "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"  # Replace with your token

    # Initialize Hugging Face authentication
    login(token=HUGGING_FACE_TOKEN)