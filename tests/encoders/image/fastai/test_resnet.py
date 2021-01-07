"""
    Test code for encoding with FastAI.
"""
from vectorhub.encoders.image.fastai import FastAIResnet2Vec
from ....test_utils import assert_encoder_works

def test_fastai_encoder():
    enc = FastAIResnet2Vec()
    assert_encoder_works(enc, 1024, data_type='image')
