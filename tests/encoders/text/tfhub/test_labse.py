from vectorhub.encoders.text.tfhub import LaBSE2Vec
import numpy as np


def test_labse_inititalize():
    """
    Testing for labse initialize
    """
    client = LaBSE2Vec()
    assert True


def test_labse_encode():
    """
    Testing for labse encode
    """
    client = LaBSE2Vec()
    result = client.encode('Cat')
    assert np.array(result).shape == (768,)
