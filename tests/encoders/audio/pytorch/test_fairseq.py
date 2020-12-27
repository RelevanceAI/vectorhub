import numpy as np
from vectorhub.encoders.audio.pytorch.wav2vec import Wav2Vec
from ....test_utils import assert_encoder_works

def test_fairseq_works():
    """
    Simple testing for Fairseq working.
    """
    enc = Wav2Vec()
    assert_model_works(enc, vector_length=512, model_type='audio')
