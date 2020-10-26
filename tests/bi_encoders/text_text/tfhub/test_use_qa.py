from vectorhub.bi_encoders.text_text.tfhub import USEMultiQA2Vec, USEQA2Vec
import numpy as np


def test_use_multi_qa_initialize():
    """
    Testing for USE-Multi-QA initialize
    """
    client = USEMultiQA2Vec()
    assert True


def test_use_multi_qa_single_encode():
    """
    Testing for USE-Multi-QA single encode
    """
    client = USEMultiQA2Vec()
    question_emb = client.encode_question('What is your age?')
    answer_emb = client.encode_answer(
        'I am 20 years old.', 'I will be 21 next year.')
    assert len(question_emb) == 512
    assert len(answer_emb) == 512


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
    Testing for USE-QA initialize
    """
    client = USEQA2Vec()
    assert True