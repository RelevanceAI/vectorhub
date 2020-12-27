import numpy as np
from vectorhub.encoders.image.tfhub import InceptionResnet2Vec
from ....test_utils import assert_encoder_works

def test_test_inception_resnet_works():
    model = InceptionResnet2Vec()
    assert_encoder_works(model, 1536, model_type='image')
