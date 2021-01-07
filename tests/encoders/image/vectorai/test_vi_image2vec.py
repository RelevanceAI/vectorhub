import os
import pytest
from vectorhub.encoders.image.vectorai import ViImage2Vec
from ....test_utils import assert_encoder_works

@pytest.mark.skip(reason="Bulk encode not implemented for ViImage2Vec")
def test_encode():
    enc = ViImage2Vec(os.environ['VI_USERNAME'], os.environ['VI_API_KEY'])
    assert_encoder_works(enc)
