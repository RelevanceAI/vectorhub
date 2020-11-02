from vectorhub.encoders.text.sentence_transformers.sentence_transformers import SentenceTransformer2Vec, LIST_OF_URLS


def test_sentence_transformers():
    """
        Sentence Transformer
    """
    for k, v in LIST_OF_URLS.items():
        enc = SentenceTransformer2Vec(model_name=k)
        assert len(enc.encode("Let us go to the beach today.")) == v['vector_length']
    