"""
    Test audio encoding
"""
from vectorhub.encoders.audio.vectorai import ViAudio2Vec
import os

def test_encode():
    enc = ViAudio2Vec(os.environ['VI_USERNAME'], os.environ['VI_API_KEY'])
    vector = enc.encode('https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_2.wav')
    assert len(vector) > 10
    