import numpy as np
from vectorhub.encoders.image.tfhub import MobileNetV12Vec, MobileNetV22Vec
from ....test_utils import assert_model_works


def test_mobilenet_model_works():
    """
    Test that mobilenet v1 works.
    """
    model = MobileNetV12Vec()
    assert_model_works(model, 1024, model_type='image')

def test_mobilenet_v2_model_works():
    """
    Test that mobilenet v2 works.
    """
    model = MobileNetV22Vec()
    assert_model_works(model, 1024, model_type='image')
