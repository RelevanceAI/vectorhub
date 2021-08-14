import pytest
from vectorhub.encoders.text.tfhub import USE2Vec

enc = USE2Vec()
enc.__name__ = "sample"

@pytest.fixture
def docs():
    return [
        {
            "text": "hey"
        },

        {
            "text": "weirdo"
        }
    ]

@pytest.fixture
def docs_with_errors():
    return [
        {
            "text": "hey"
        },

        {
            "text": None
        }
    ]

def assert_vectors_in_docs(docs):
    for d in docs:
        assert "text_sample_vector_" in d, "misssing vector"

def test_encode_documents_in_docs(docs):
    docs = enc.encode_documents(["text"], docs)
    assert_vectors_in_docs(docs)

def test_encode_documents_in_docs_2(docs):
    docs = enc.encode_documents_in_bulk(["text"], docs, 
        vector_error_treatment="zero_vector")
    assert_vectors_in_docs(docs)

def test_encode_documents_in_docs_3(docs):
    docs = enc.encode_documents_in_bulk(["text"], docs, 
        vector_error_treatment="do_not_include")
    assert_vectors_in_docs(docs)

def test_error_tests(docs_with_errors):
    docs = enc.encode_documents(["text"], docs_with_errors, 
        vector_error_treatment="zero_vector")
    assert_vectors_in_docs(docs)

def test_error_tests_2(docs_with_errors):
    docs = enc.encode_documents_in_bulk(["text"], docs_with_errors, 
        vector_error_treatment="zero_vector")
    assert_vectors_in_docs(docs)

def test_error_tests_3(docs_with_errors):
    docs = enc.encode_documents_in_bulk(["text"], docs_with_errors, 
        vector_error_treatment="do_not_include")
    assert "text_sample_vector_" in docs[0]
    assert "text_sample_vector_" not in docs[-1]
    assert isinstance(docs[0]['text_sample_vector_'], list)
