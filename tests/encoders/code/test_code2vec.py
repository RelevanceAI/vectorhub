"""
Face 2 Vec
"""
import numpy as np
from vectorhub.encoders.code.transformers import Code2Vec
from ...test_utils import assert_encoder_works

def test_code_2_vec_works():
    """
    Testing FaceNet works
    """
    model = Code2Vec()
    assert_encoder_works(model, 768, data_type='text')
