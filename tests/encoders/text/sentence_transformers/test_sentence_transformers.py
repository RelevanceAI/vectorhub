from vectorhub.encoders.text.sentence_transformers.sentence_auto_transformers import SentenceTransformer2Vec, LIST_OF_URLS
import gc

def test_sentence_transformers():
    """
        Sentence Transformer
    """
    for k, v in LIST_OF_URLS.items():
        enc = SentenceTransformer2Vec(model_name=k)
        assert len(enc.encode("Let us go to the beach today.")) == v['vector_length']
        assert len(enc.bulk_encode(["hi", "whats up"])) == 2
        del enc; gc.collect();
