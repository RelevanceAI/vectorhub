import numpy as np
from vectorhub.encoders.audio.tfhub import Trill2Vec, TrillDistilled2Vec

def test_trill_initialize():
    """
    Testing for the trill successfull trill initialize
    """

    client = Trill2Vec()
    assert True


def test_trill_encode():
    """
    Testing for the trill single encode
    """

    client = Trill2Vec()
    sample = client.read(
        'https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_2.wav', 16000)

    embedding = client.encode(sample)
    assert len(embedding) == 512


def test_trill_bulk_encode():
    """
    Testing for the trill bulk encode
    """

    client = Trill2Vec()
    list_of_audio = [
        'https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_2.wav',
        'https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_1.wav'
    ]
    samples = [client.read(audio) for audio in list_of_audio]
    multi_embedding = client.bulk_encode(samples)
    assert len(multi_embedding[0]) == 512
    assert len(multi_embedding) == 2


def test_trill_distlled_initilize():
    """
    Testing for the trill distlled initialize
    """

    client = TrillDistilled2Vec()
    assert True


def test_trill_distlled_encode():
    """
    Testing for the trill distlled single encode
    """
    client = TrillDistilled2Vec()
    sample = client.read(
        'https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_2.wav', 16000)

    embedding = client.encode(sample)
    assert len(embedding) == 2048


def test_trill_distlled_bulk_encode():
    """
    Testing for the trill distlled bulk encode
    """
    client = TrillDistilled2Vec()
    list_of_audio = [
        'https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_2.wav',
        'https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_1.wav'
    ]
    samples = [client.read(audio) for audio in list_of_audio]
    multi_embedding = client.bulk_encode(samples)
    assert len(multi_embedding[0]) == 2048
    assert len(multi_embedding) == 2
