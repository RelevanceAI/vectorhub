import gc
from vectorhub.encoders.text.sentence_transformers.sentence_auto_transformers import SentenceTransformer2Vec, LIST_OF_URLS
from ....test_utils import assert_encoder_works

def test_sentence_transformers():
    """
        Sentence Transformer
    """
    for k, v in LIST_OF_URLS.items():
        enc = SentenceTransformer2Vec(model_name=k)
        assert_encoder_works(enc, model_type='text')
        del enc; gc.collect();
