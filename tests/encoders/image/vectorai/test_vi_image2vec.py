from vectorhub.encoders.image.vectorai import ViImage2Vec
from ....test_utils import assert_encoder_works

def test_encode():
    enc = ViImage2Vec(os.environ['VI_USERNAME'], os.environ['VI_API_KEY'])
    assert_encoder_works(enc)
