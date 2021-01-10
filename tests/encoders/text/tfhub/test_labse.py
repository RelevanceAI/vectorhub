import numpy as np
from vectorhub.encoders.text.tfhub import LaBSE2Vec
from ....test_utils import assert_encoder_works

def test_labse_encode():
    """
    Testing for labse encode
    """
    import tensorflow as tf
    if hasattr(tf, 'executing_eagerly'):
        if not tf.executing_eagerly():
            tf.compat.v1.enable_eager_execution()
    encoder = LaBSE2Vec()
    assert_encoder_works(encoder, vector_length=768, data_type='text')
