"""
    Test code for encoding with FastAI.
"""

from vectorhub.encoders.image.fastai import FastAIResnet2Vec

def test_fastai_encoder():
    enc = FastAIResnet2Vec()
    arr = enc.encode('https://getvectorai.com/assets/logo-square.png')
    assert len(arr) == 1024
