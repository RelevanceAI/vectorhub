import os
from vectorhub.encoders.text.vectorai import ViText2Vec

def test_encoder():
    enc = ViText2Vec(os.environ['VI_USERNAME'], os.environ['VI_API_KEY'])
    vector = enc.encode("HI")
    assert len(vector) > 10
    vectors = enc.bulk_encode(["Hey", "Stranger!"])
    assert len(vectors) == 2
