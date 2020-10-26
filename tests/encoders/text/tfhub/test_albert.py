from vectorhub.encoders.text.tfhub import Albert2Vec
import numpy as np


def test_albert_initialize():
    """
    Testing for albert initialize
    """
    client = Albert2Vec()
    assert True


def test_albert_encode():
    """
    Testing for albert initialize
    """
    client = Albert2Vec()
    result = client.encode('Cat')
    assert np.array(result).shape == (768,)
