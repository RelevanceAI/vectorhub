from vectorhub.encoders.text.tfhub import Bert2Vec
import numpy as np


def test_bert_initialize():
    """
    Testing for initialize bert model
    """
    client = Bert2Vec()
    assert True


def test_bert_encode():
    """
    Testing for bert encoding
    """
    client = Bert2Vec()
    result = client.encode('Cat')
    assert np.array(result).shape == (1024,)
