from vectorhub.encoders.audio.pytorch.fairseq import Wav2Vec
import numpy as np


def test_fairseq_initialize():
    """
    Testing for fairseq initialize
    """

    client = Wav2Vec()
    assert True


def test_fairseq_encoder():
    """
    Testing for fairseq single encode
    """
    client = Wav2Vec()
    audio = client.read(
        'https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_2.wav')
    output = client.encode(audio)
    assert len(output) == 512
