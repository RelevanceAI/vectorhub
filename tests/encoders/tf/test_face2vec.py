"""
Face 2 Vec
"""
import numpy as np
from vectorhub.encoders.face.tf import Face2Vec
from ....test_utils import assert_encoder_works

def test_face_2_vec_works():
    """
    Testing FaceNet works
    """
    model = Face2Vec()
    assert_encoder_works(model, 128, data_type='image', 
    image_url='https://www.thestatesman.com/wp-content/uploads/2017/08/1493458748-beauty-face-517.jpg')
