import numpy as np
from vectorhub.encoders.text.tfhub import USETransformer2Vec
from ....test_utils import assert_encoder_works

def test_labse_encode():
    """
    Testing for USE encode
    """
    import tensorflow as tf
    encoder = USETransformer2Vec()
    assert_encoder_works(encoder, vector_length=1024, data_type='text')

def test_access_urls():
    """Test Access to the URLS.
    """
    urls = USETransformer2Vec.urls
    assert isinstance(urls, dict)
