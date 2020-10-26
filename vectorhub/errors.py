"""Errors
"""

class ModelError(Exception):
    """Base error class for all errors in library
    """

    def __init__(self, message: str):
        """
        The main Vector Hub base error.
        Args:
            message: The error message
        Example:
            >>> raise ModelError("Missing ____.")
        """
        self.response_message = message