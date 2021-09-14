import pytest
from vectorhub.encoders.text.tfhub import USE2Vec

enc = USE2Vec()
enc.__name__ = "sample"

@pytest.fixture
def chunk_docs():
    return [{
        "value": [
            {
                "text": "hey"
            },

            {
                "text": "weirdo"
            }
        ]},
        {"value": [
            {
                "text": "hello"
            },

            {
                "text": "stranger"
            }
        ]},
    ]

def assert_vectors_in_docs(docs):
    for d in docs:
        assert "text_sample_chunkvector_" in d['value'][0], "misssing vector"

def test_encode_documents_in_docs(chunk_docs):
    chunk_docs = enc.encode_chunk_documents(chunk_field="value", fields=["text"], documents=chunk_docs)
    assert_vectors_in_docs(chunk_docs)

