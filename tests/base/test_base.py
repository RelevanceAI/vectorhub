"""
    Tests for various base functions occur here.
"""

import numpy as np 
from vectorhub.base import catch_vector_errors
from vectorhub.encoders.text.torch_transformers import Transformer2Vec

def test_catch_vector_errors():
    """Test the catch vector errors.
    """
    encoder = Transformer2Vec('bert-base-uncased')
    vectors = encoder.encode(np.nan)
    assert len(vectors) == 768
    assert vectors[0] == 1e-7
