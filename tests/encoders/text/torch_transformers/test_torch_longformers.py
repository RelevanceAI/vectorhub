"""
    Testing for Longformers with Torch requirement
"""
import pytest
import numpy as np
from vectorhub.encoders.text.torch_transformers import Longformer2Vec
from ....test_utils import assert_encoder_works 

MODEL_LIST = [
    "allenai/longformer-base-4096",
    "allenai/longformer-large-4096",
]

VECTOR_OUTPUT = {
    "allenai/longformer-base-4096": 768,
    "allenai/longformer-large-4096": 1024,
}

@pytest.mark.skip(reason="Model too big.")
@pytest.mark.parametrize("model_name", MODEL_LIST)
def test_torch_transformer_encode(model_name):
    model = Longformer2Vec(model_name)
    assert_encoder_works(model, vector_Length=VECTOR_OUTPUT[model_name], data_type='text')
