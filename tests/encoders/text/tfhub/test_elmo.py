import pytest
from vectorhub.encoders.text.tfhub import Elmo2Vec
from ....test_utils import assert_encoder_works 

def test_elmo_encode():
    """
    Testing for Elmo encoding
    """
    enc = Elmo2Vec()
    assert_encoder_works(enc, 1024, model_type='text')

@pytest.mark.parametrize('output_layer', ['lstm_outputs1','lstm_outputs2', 'default'])
def test_all_elmo_encoding_methods(output_layer):
    """
    Check that the elmo signatures work.
    """
    enc = Elmo2Vec()
    assert_encoder_works(enc, vector_length=1024, model_type='text')
