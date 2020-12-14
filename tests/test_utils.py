import numpy as np
from vectorhub.utils import *

def test_list_models():
    assert len(list_models()) > 0

def test_list_installed_models():
    # Vector AI deployed models should be immediately usable
    assert len(list_installed_models()) > 0
    
def is_dummy_vector(vector, vector_length=None):
    """
        Return True if the vector is the default vector, False if it is not.
    """
    if vector_length is None:
        vector_length = len(vector)
    return vector == [1e-7] * vector_length

def assert_vector_works(vector, vector_length=None):
    """
        Assert that the vector works as intended.
    """
    assert isinstance(vector, list), "Not the right data type - needs to be a list!"
    assert not is_dummy_vector(vector, vector_length),  "Is a dummy vector"
    if vector_length is not None:
        assert len(vector) == vector_length, f"Does not match vector length of {vector_length}"
