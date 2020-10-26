from vectorhub.encoders.image.tfhub import ResnetV12Vec, ResnetV22Vec
import numpy as np


def test_resnet_v1_initialize():
    """
    Testing for resnet v1 initialize
    """
    client = ResnetV12Vec()
    assert True


def test_resnet_v1_encode():
    """
    Testing for resnet v1 encode
    """
    client = ResnetV12Vec()
    sample = client.read('https://getvectorai.com/assets/logo-square.png')
    result = client.encode(sample)
    assert np.array(result).shape == (2048,)


def test_resnet_v1_bulk_encode():
    """
    Testing for resnet v1 bulk_encode
    """
    client = ResnetV12Vec()
    sample = client.read('https://getvectorai.com/assets/logo-square.png')
    result = client.bulk_encode([sample, sample])
    assert np.array(result).shape == (2, 2048)


def test_resnet_v2_initialize():
    """
    Testing for resnet v2 initialize
    """
    client = ResnetV22Vec()
    assert True
