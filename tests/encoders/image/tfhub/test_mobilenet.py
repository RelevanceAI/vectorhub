import numpy as np
from vectorhub.encoders.image.tfhub import MobileNetV12Vec, MobileNetV22Vec
from ....test_utils import assert_vector_works
from ....test_utils import AssertModelWorks

def test_mobilenet_model_works():
    """
    Test that mobilenet v1 works.
    """
    model = MobileNetV12Vec()
    model_check = AssertModelWorks(model, 1024, model_type='image')
    model_check.assert_encoding_methods_work()

def test_mobilenet_v2_model_works():
    """
    Test that mobilenet v2 works.
    """
    model = MobileNetV22Vec()
    model_check = AssertModelWorks(model, 1024, model_type='image')
    model_check.assert_encoding_methods_work()
