import numpy as np
from vectorhub.encoders.audio.tfhub.yamnet import Yamnet2Vec
from ....test_utils import assert_encoder_works


def test_yamnet_initialize():
    """
    Testing for the yamnet initialize
    """
    model = Yamnet2Vec()
    assert_encoder_works(model, vector_length=1024, model_type='audio')
