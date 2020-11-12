from vectorhub.encoders.text.tfhub import Bert2Vec
import numpy as np

def test_bert_encode():
    """
    Testing for bert encoding
    """
    client = Bert2Vec('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3')
    result = client.encode('Cat')
    assert np.array(result).shape == (1024,)
