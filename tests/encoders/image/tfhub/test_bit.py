import numpy as np
from vectorhub.encoders.image.tfhub import BitMedium2Vec, BitSmall2Vec
from ....test_utils import assert_model_works

def test_bit_medium_works():
    """
    Testing BIT medium works
    """
    model = BitMedium2Vec()
    assert_model_works(model, 2048, model_type='image')

def test_bit_small_works():
    """
    Testing BIT small works
    """
    model = BitSmall2Vec()
    assert_model_works(model, 2048, model_type='image')
