import numpy as np
from vectorhub.encoders.image.tfhub import MobileNetV12Vec, MobileNetV22Vec
from ....test_utils import assert_encoder_works


def test_mobilenet_model_works():
    """
    Test that mobilenet v1 works.
    """
    model = MobileNetV12Vec()
    assert_encoder_works(model, 1024, data_type='image')

def test_mobilenet_v2_model_works():
    """
    Test that mobilenet v2 works.
    """
    model = MobileNetV22Vec()
    assert_encoder_works(model, 1792, data_type='image')
