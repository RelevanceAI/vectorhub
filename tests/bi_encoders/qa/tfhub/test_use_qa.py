import numpy as np
from vectorhub.bi_encoders.qa.tfhub import USEMultiQA2Vec, USEQA2Vec
from ....test_utils import assert_encoder_works

def test_use_multi_qa_initialize():
    """
    Testing for USE-Multi-QA initialize
    """
    encoder = USEMultiQA2Vec()
    assert_encoder_works(encoder, data_type='text', model_type='bi_encoder')

def test_use_multi_qa_single_encode():
    """
    Testing for USE-Multi-QA single encode
    """
    encoder = USEMultiQA2Vec()
    assert_encoder_works(encoder, data_type='text', model_type='bi_encoder')

def test_use_multi_qa_bulk_encode():
    """
    Testing for USE-Multi-QA bulk encode
    """
    client = USEMultiQA2Vec()
    question_emb = client.bulk_encode_questions(['What is your age?'])
    answer_emb = client.bulk_encode_answers(["I am 20 years old.", "good morning"], [
                                           "I will be 21 next year.", "great day."])
    assert len(question_emb) == 1
    assert len(answer_emb) == 2


def test_use_qa_initialize():
    """
    Testing for USE-QA.
    """
    encoder = USEQA2Vec()
    assert_encoder_works(encoder, vector_length=512, data_type='text', model_type='bi_encoder')
