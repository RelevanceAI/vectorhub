import numpy as np
from vectorhub.encoders.text.tfhub import Albert2Vec
from ....test_utils import assert_encoder_works 

def test_albert_encode():
    """
    Testing for albert initialize
    """
    enc = Albert2Vec()
    assert_encoder_works(enc, 768, 'text')
