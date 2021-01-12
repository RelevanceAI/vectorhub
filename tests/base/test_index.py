"""
    Tests for various base functions occur here.
"""
import pytest
import os
from vectorhub.encoders.audio.tfhub import SpeechEmbedding2Vec

class TestIndex:
    """
    Testing the ability to use and add to the Vector AI index.
    """
    def test_vi_index(audio_url):
        num_of_documents = 30
        enc = SpeechEmbedding2Vec()
        items = [audio_url] * num_of_documents
        response = enc.add_documents(
            os.environ['VH_USERNAME'],
            os.environ['VH_API_KEY'],
            items=items,
            collection_name='test_index')
        assert response['successfully_inserted'] == num_of_documents
        enc.client.delete_collection(enc.collection_name)

    def test_vi_index_with_metadata(audio_url):
        """
        Test the Vector AI index with Metadata.
        """
        num_of_documents = 30
        enc = SpeechEmbedding2Vec()
        items= [audio_url] * num_of_documents
        metadata = list(range(num_of_documents))
        response = enc.add_documents(
            os.environ['VH_USERNAME'],
            os.environ['VH_API_KEY'],
            items=items,
            metadata=metadata,
            collection_name='test_index')
        assert response['successfully_inserted'] == num_of_documents
        enc.client.delete_collection(enc.collection_name)

