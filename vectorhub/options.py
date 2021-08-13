"""
Python Options
To access Python options for VectorHub
"""
from enum import Enum
class IfErrorReturns(Enum):
    RETURN_NONE: str="RETURN_NONE"
    RETURN_EMPTY_VECTOR: str = "RETURN_EMPTY_VECTOR"
    RAISE_ERROR: str = "RAISE_ERROR"
    
OPTIONS = {
    'if_error': IfErrorReturns.RETURN_NONE
}

def get_option(field):
    """Get an option with a specific field name.
    """
    return OPTIONS[field]

def set_option(field, value):
    """Set an option with a specific field name.
    """
    OPTIONS[field] = value
