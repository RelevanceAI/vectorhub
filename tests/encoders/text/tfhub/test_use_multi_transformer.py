import numpy as np
from vectorhub.encoders.text.tfhub import USEMultiTransformer2Vec
from ....test_utils import assert_encoder_works

def test_use_multi_transformer():
    """
    Testing for USE encode
    """
    import tensorflow as tf
    encoder = USEMultiTransformer2Vec()
    assert_encoder_works(encoder, vector_length=1024, data_type='text')

