from vectorhub.bi_encoders.text_text.tfhub import LAReQA2Vec

def test_lare_qa_initialize():
    """
    Testing for LAReQA initialize
    """
    client = LAReQA2Vec()
    assert True