import numpy as np
from vectorhub.encoders.image.tfhub import ResnetV12Vec, ResnetV22Vec
from ....test_utils import assert_model_works

def test_resnet_v1_works():
    """
    Test that mobilenet v2 works.
    """
    model = ResnetV12Vec()
    assert_model_works(model, 2048, 'image')

def test_resnet_v2_initialize():
    """
    Testing for resnet v2 initialize
    """
    model = ResnetV22Vec()
    assert_model_works(model, 2048, 'image')
