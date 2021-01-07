"""
    Testing for Transformers with TF requirement
"""
import pytest
from vectorhub.encoders.text.tf_transformers import TFTransformer2Vec
from ....test_utils import assert_encoder_works 

@pytest.mark.parametrize("model_name",["distilbert-base-uncased"])
def test_tf_transformer_encode(model_name):
    """
        Test for encoding transformer models
    """
    model = TFTransformer2Vec(model_name)
    assert_encoder_works(model, data_type='text')
