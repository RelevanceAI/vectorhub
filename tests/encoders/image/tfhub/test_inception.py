import numpy as np
from vectorhub.encoders.image.tfhub import InceptionV12Vec, InceptionV22Vec, InceptionV32Vec
from ....test_utils import assert_encoder_works

def test_inception_v1_works():
    """
    Test that mobilenet v1 works.
    """
    model = InceptionV12Vec()
    assert_encoder_works(model, 1024, data_type='image')

def test_inception_v2_works():
    """
    Test that mobilenet v1 works.
    """
    model = InceptionV22Vec()
    assert_encoder_works(model, 1024, data_type='image')

def test_inception_v3_works():
    """
    Testing for inception v3 initialize
    """
    model = InceptionV32Vec()
    assert_encoder_works(model, 2048, data_type='image')
