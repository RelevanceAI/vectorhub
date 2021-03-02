"""
    Tests for various base functions occur here.
"""
import pytest
import numpy as np 
import os
import vectorhub
from vectorhub.base import catch_vector_errors, Base2Vec
from vectorhub.encoders.audio.tfhub import SpeechEmbedding2Vec
from ..test_utils import is_dummy_vector

def test_catch_vector_errors():
    """Test the catch vector errors.
    """
    encoder = SpeechEmbedding2Vec()
    vectors = encoder.encode(np.nan)
    assert is_dummy_vector(vectors)

def test_catch_vector_errors_false():
    """Test catch the vector errors
    """
    with pytest.raises(Exception):
        vectorhub.options.set_option('catch_vector_error', False)
        encoder = SpeechEmbedding2Vec()
        vectors = encoder.encode(np.nan)

def test_validate_urls_raises_warning():
    enc = Base2Vec()
    with pytest.warns(UserWarning):
        # Assert this is false
        assert not enc.validate_model_url('testing_url', ['testing_url_2', 'fake_url'])

def test_validate_urls_works_for_tfhub_exception():
    enc = Base2Vec()
    assert enc.validate_model_url('https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/3', 
    ['https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1', 'https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/2'])

def test_validate_urls_works_simple():
    enc = Base2Vec()
    assert enc.validate_model_url('test', ['test', 'test_2'])
