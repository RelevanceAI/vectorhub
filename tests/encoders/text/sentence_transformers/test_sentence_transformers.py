import gc
import pytest
from vectorhub.encoders.text.sentence_transformers.sentence_auto_transformers import SentenceTransformer2Vec, LIST_OF_URLS
from ....test_utils import assert_encoder_works

@pytest.mark.skip("URL errors with WSL linux containers.")
@pytest.mark.parametrize("model_name", list(LIST_OF_URLS.keys()))
def test_sentence_transformers(model_name):
    """
        Sentence Transformer
    """
    enc = SentenceTransformer2Vec(model_name=model_name)
    assert_encoder_works(enc, data_type='text')
    gc.collect();
