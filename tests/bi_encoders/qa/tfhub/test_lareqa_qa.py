from vectorhub.bi_encoders.qa.tfhub import LAReQA2Vec
from ....test_utils import assert_encoder_works

def test_lare_qa_works():
    """
    Testing for LAReQA works
    """
    encoder = LAReQA2Vec()
    assert_encoder_works(encoder, data_type='text', model_type='bi_encoder')
