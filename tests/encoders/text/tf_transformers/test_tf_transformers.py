"""
    Testing for Transformers with TF requirement
"""
import pytest
from vectorhub.encoders.text.tf_transformers import TFTransformer2Vec

@pytest.mark.parametrize("model_name",["bert-base-uncased", "distilbert-base-uncased"])
def test_tf_transformer_encode(model_name):
    """
        Test for encoding transformer models
    """
    model = Transformer2Vec(model_name)
    vector = model.encode("Hi!")
    assert len(vector) > 0

@pytest.mark.parametrize("model_name",["bert-base-uncased", "distilbert-base-uncased"])
def test_tf_transformer_bulk_encode(model_name):
    """
        Test for bulk encoding.
    """
    model = Transformer2Vec(model_name)
    vector = model.bulk_encode(["Hi!", "there"])
    assert len(vector) == 2
