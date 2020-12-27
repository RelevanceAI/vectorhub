import numpy as np
from vectorhub.encoders.text.tfhub import LaBSE2Vec
from ....test_utils import assert_encoder_works

def test_labse_encode():
    """
    Testing for labse encode
    """
    encoder = LaBSE2Vec()
    assert_encoder_works(encoder, vector_length=768, model_type='text')
