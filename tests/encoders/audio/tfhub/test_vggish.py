import numpy as np
from vectorhub.encoders.audio.tfhub.vggish import Vggish2Vec
from ....test_utils import assert_encoder_works

def test_vggish_initialize():
    """
    Testing for the vggish initialize
    """
    model = Vggish2Vec()
    assert_encoder_works(model, vector_length=128, model_type='audio')
