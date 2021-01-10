"""
    Testing for Longformers with Torch requirement
"""
import pytest
import numpy as np
from vectorhub.encoders.text.torch_transformers import LegalBert2Vec
from ....test_utils import assert_encoder_works 

def test_torch_transformer_encode():
    model = LegalBert2Vec()
    assert_encoder_works(model, data_type='text')
