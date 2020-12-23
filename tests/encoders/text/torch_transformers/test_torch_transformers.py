"""
    Testing for Transformers with Torch requirement
"""
import pytest
from vectorhub.encoders.text.torch_transformers import Transformer2Vec

MODEL_LIST = [
    "bert-base-uncased", 
    "distilbert-base-uncased", 
    "facebook/bart-base"
]

@pytest.mark.parametrize("model_name", MODEL_LIST)
def test_torch_transformer_encode(model_name):
    model = Transformer2Vec(model_name)
    vector = model.encode("Hi!")
    assert_vector_works(vector)
    assert len(vector) > 0

@pytest.mark.parametrize("model_name", MODEL_LIST)
def test_torch_transformer_bulk_encode(model_name):
    model = Transformer2Vec(model_name)
    vector = model.bulk_encode(["Hi!", "there"])
    assert len(vector) == 2
    assert_vector_works(vector)
