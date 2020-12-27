import numpy as np
from vectorhub.encoders.audio.tfhub import Trill2Vec, TrillDistilled2Vec
from ....test_utils import assert_encoder_works

def test_trill_works():
    """
    Testing for speech embedding initialization
    """
    enc = Trill2Vec()
    assert_encoder_works(enc, vector_length=512, data_type='audio')

def test_trill_distilled_works():
    enc = TrillDistilled2Vec()
    assert_encoder_works(enc, vector_length=2048, data_type='audio')
