from vectorhub.encoders.audio.tfhub.vggish import Vggish2Vec
import numpy as np


def test_vggish_initialize():
    """
    Testing for the vggish initialize
    """
    client = Vggish2Vec()
    assert True


def test_vggish_encode():
    """
    Testing for vggish single encode
    """
    client = Vggish2Vec()
    sample = client.read(
        'https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_2.wav', 16000)

    embedding = client.encode(sample)
    assert len(embedding) == 128


def test_vggish_bulk_encode():
    """
    Testing for vggish bulk encode
    """
    client = Vggish2Vec()
    list_of_audio = [
        'https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_2.wav',
        'https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_1.wav'
    ]
    samples = [client.read(audio) for audio in list_of_audio]
    multi_embedding = client.bulk_encode(samples)
    assert len(multi_embedding[0]) == 128
    assert len(multi_embedding) == 2
