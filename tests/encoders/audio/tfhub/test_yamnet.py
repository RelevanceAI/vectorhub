from vectorhub.encoders.audio.tfhub.yamnet import Yamnet2Vec
import numpy as np


def test_yamnet_initialize():
    """
    Testing for the yamnet initialize
    """
    client = Yamnet2Vec()
    assert True


def test_yamnet_encode():
    """
    Testing for yamnet single encode
    """
    client = Yamnet2Vec()
    sample = client.read(
        'https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_2.wav', 16000)

    embedding = client.encode(sample)
    assert len(embedding) == 1024


def test_yamnet_bulk_encode():
    """
    Testing for yamnet bulk encode
    """
    client = Yamnet2Vec()
    list_of_audio = [
        'https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_2.wav',
        'https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_1.wav'
    ]
    samples = [client.read(audio) for audio in list_of_audio]
    multi_embedding = client.bulk_encode(samples)
    assert len(multi_embedding[0]) == 1024
    assert len(multi_embedding) == 2
