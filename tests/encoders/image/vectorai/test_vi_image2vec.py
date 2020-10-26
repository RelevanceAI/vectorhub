from vectorhub.encoders.image.vectorai import ViImage2Vec
import os

def test_encode():
    enc = ViImage2Vec(os.environ['VI_USERNAME'], os.environ['VI_API_KEY'])
    vector = enc.encode('https://getvectorai.com/assets/logo-square.png')
    assert len(vector) > 10
