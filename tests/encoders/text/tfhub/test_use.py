from vectorhub.encoders.text.tfhub import USE2Vec, USEMulti2Vec, USELite2Vec
from ....test_utils import assert_encoder_works

def test_use_encode():
    """
    Testing for labse encode
    """
    encoder = USE2Vec()
    assert_encoder_works(encoder, vector_length=512, model_type='text')

def test_use_multi_encode():
    """
    Testing for labse encode
    """
    encoder = USEMulti2Vec()
    assert_encoder_works(encoder, vector_length=512, model_type='text')

def test_use_lite_works():
    """
    Testing for USE encoder
    """
    encoder = USELite2Vec()
    assert_encoder_works(encoder, vector_length=512, model_type='text')
