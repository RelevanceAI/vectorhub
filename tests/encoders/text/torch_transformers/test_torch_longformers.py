"""
    Testing for Longformers with Torch requirement
"""
import pytest
import numpy as np
from vectorhub.encoders.text.torch_transformers import Longformer2Vec
from ....test_utils import assert_vector_works

MODEL_LIST = [
    "allenai/longformer-base-4096",
    "allenai/longformer-large-4096",
]

VECTOR_OUTPUT = {
    "allenai/longformer-base-4096": 768,
    "allenai/longformer-large-4096": 1024,
}


@pytest.mark.parametrize("model_name", MODEL_LIST)
def test_torch_transformer_encode(model_name):
    model = Longformer2Vec(model_name)
    vector = model.encode("I enjoy taking long walks along the beach with my dog.")
    assert_vector_works(vector, VECTOR_OUTPUT[model_name])

@pytest.mark.parametrize("model_name", MODEL_LIST)
def test_torch_transformer_bulk_encode(model_name):
    model = Longformer2Vec(model_name)
    vector = model.bulk_encode(["Hi!", "there"])
    assert_vector_works(vector, )
    assert np.array(vector).shape == (2, VECTOR_OUTPUT[model_name])
    assert_vector_works(vector, VECTOR_OUTPUT[model_name])
