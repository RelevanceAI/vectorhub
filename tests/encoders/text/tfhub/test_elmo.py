import pytest
from vectorhub.encoders.text.tfhub import Elmo2Vec
from ....test_utils import assert_vector_works

def test_elmo_encode():
    """
    Testing for Elmo encoding
    """
    enc = Elmo2Vec()
    vector = enc.encode('Cat')
    assert_vector_works(vector, 1024)

def test_elmo_bulk_encode():
    """
    Testing for Elmo encoding
    """
    enc = Elmo2Vec()
    result = enc.bulk_encode(['Cat', 'Waves'])
    print(len(result))
    print(len(result[0]))
    assert len(result) == 2
    assert len(result[0]) == 1024

@pytest.mark.parametrize('output_layer', ['lstm_outputs1','lstm_outputs2', 'default'])
def test_all_elmo_encoding_methods(output_layer):
    """
    Check that the elmo signatures work.
    """
    enc = Elmo2Vec()
    vector = enc.encode('Cat', output_layer=output_layer)
    assert_vector_works(vector)

@pytest.mark.parametrize('output_layer', ['lstm_outputs1','lstm_outputs2', 'default'])
def test_all_elmo_bulk_encoding_methods(output_layer):
    """
    Check that the elmo signatures work.
    """
    enc = Elmo2Vec()
    vectors = enc.bulk_encode(['Cat', "waves", "Hi!"], output_layer=output_layer)
    assert len(vectors) == 3
    for vec in vectors:
        assert_vector_works(vec)
