import numpy as np
from vectorhub.encoders.image.tfhub import InceptionV12Vec, InceptionV22Vec, InceptionV32Vec


def test_inception_v1_initialize():
    """
    Testing for inception v1 initialize
    """
    client = InceptionV12Vec()
    assert True


def test_inception_v1_encode():
    """
    Testing for inception v1 encode
    """
    client = InceptionV12Vec()
    sample = client.read('https://getvectorai.com/assets/logo-square.png')
    result = client.encode(sample)
    assert np.array(result).shape == (1024,)


def test_inception_v1_bulk_encode():
    """
    Testing for inception v1 bulk encode
    """
    client = InceptionV12Vec()
    sample = client.read('https://getvectorai.com/assets/logo-square.png')
    result = client.bulk_encode([sample, sample])
    assert np.array(result).shape == (2, 1024)


def test_inception_v2_initialize():
    """
    Testing for inception v2 initialize
    """
    client = InceptionV22Vec()
    assert True


def test_inception_v3_initialize():
    """
    Testing for inception v3 initialize
    """
    client = InceptionV32Vec()
    assert True
