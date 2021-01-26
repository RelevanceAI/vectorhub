from vectorhub.bi_encoders.text_image.torch import Clip2Vec
from ....test_utils import assert_encoder_works

def test_lare_qa_works():
    """
    Testing for LAReQA works
    """
    encoder = LAReQA2Vec()
    assert_encoder_works(encoder, data_type='text_image', model_type='bi_encoder')
