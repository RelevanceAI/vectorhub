"""
    Testing for Transformers with Torch requirement
"""
import pytest
from vectorhub.encoders.text.torch_transformers import Transformer2Vec
from ....test_utils import assert_encoder_works 

MODEL_LIST = [
    "bert-base-uncased", 
    "distilbert-base-uncased", 
    "facebook/bart-base"
]

# TODO: Add vector output into the model name and type

@pytest.mark.parametrize("model_name", MODEL_LIST)
def test_torch_transformer_encode(model_name):
    model = Transformer2Vec(model_name)
    assert_encoder_works(model, model_type='text')
