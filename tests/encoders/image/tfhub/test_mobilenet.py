from vectorhub.encoders.image.tfhub import MobileNetV12Vec, MobileNetV22Vec
import numpy as np


def test_mobilenet_v1_initialize():
    """
    Testing for mobilenet v1 initialize
    """
    client = MobileNetV12Vec()
    assert True


def test_mobilenet_v1_encode():
    """
    Testing for mobilenet v1 encode
    """
    client = MobileNetV12Vec()
    sample = client.read('https://getvectorai.com/assets/logo-square.png')
    result = client.encode(sample)
    assert np.array(result).shape == (1024,)


def test_mobilenet_v1_bulk_encode():
    """
    Testing for mobilenet v1 bulk encode
    """
    client = MobileNetV12Vec()
    sample = client.read('https://getvectorai.com/assets/logo-square.png')
    result = client.bulk_encode([sample, sample])
    assert np.array(result).shape == (2, 1024)

def test_mobilenet_v2_initialize():
    """
    Testing for mobilenet v2 initialize
    """
    client = MobileNetV22Vec()
    assert True