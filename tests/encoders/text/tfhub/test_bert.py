import numpy as np
from vectorhub.encoders.text.tfhub import Bert2Vec
from ....test_utils import assert_encoder_works

def test_bert_encode():
    """
    Testing for bert encoding
    """
    encoder = Bert2Vec()
    assert_encoder_works(encoder, vector_length=1024, model_type='text')
