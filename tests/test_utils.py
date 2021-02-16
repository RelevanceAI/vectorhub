import numpy as np
import os
import random
import time
import string
import gc
from vectorhub.utils import list_models, list_installed_models
from vectorai import ViClient, ViCollectionClient

class TempClient:
    """Client For a temporary collection
    """
    def __init__(self, client, collection_name: str=None):
        if client is None: 
            raise ValueError("Client cannot be None.")
        self.client = client
        if isinstance(client, ViClient):
            self.collection_name = collection_name
        elif isinstance(client, ViCollectionClient):
            self.collection_name = self.client.collection_name
        else:
            self.collection_name = collection_name

    def teardown_collection(self):
        if self.collection_name in self.client.list_collections():
            time.sleep(2)
            if isinstance(self.client, ViClient):
                self.client.delete_collection(self.collection_name)
            elif isinstance(self.client, ViCollectionClient):
                self.client.delete_collection()
    
    def __enter__(self):
        self.teardown_collection()
        return self.client
    
    def __exit__(self, *exc):
        self.teardown_collection()


def test_list_models():
    assert len(list_models()) > 0

def test_list_installed_models():
    # Vector AI deployed models should be immediately usable
    assert len(list_installed_models()) > 0
    
def is_dummy_vector(vector, vector_length=None):
    """
        Return True if the vector is the default vector, False if it is not.
    """
    if vector_length is None:
        vector_length = len(vector)
    return vector == [1e-7] * vector_length

def assert_vector_works(vector, vector_length=None):
    """
        Assert that the vector works as intended.
    """
    assert isinstance(vector, list), "Not the right data type - needs to be a list!"
    # Assert that the vector works if this is in bulk
    if isinstance(vector[0], list):
        # If this is a list of vectors as opposed to just one
        for v in vector:
            assert not is_dummy_vector(vector[0], vector_length),  "Is a dummy vector"
            if vector_length is not None:
                assert len(vector[0]) == vector_length, f"Does not match vector length of {vector_length}"
    else:
        # Assert vector works if it is just 1 vector.
        assert not is_dummy_vector(vector, vector_length),  "Is a dummy vector"
        if vector_length is not None:
            assert len(vector) == vector_length, f"Does not match vector length of {vector_length}"

class AssertModelWorks:
    def __init__(self, model, vector_length, data_type='image', model_type='encoder', 
    image_url: str='https://getvectorai.com/_nuxt/img/dog-1.3cc5fe1.png',
    audio_url: str='https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_2.wav',
    sample_sentence: str= "Cats enjoy purring in the nature.",
    sample_question: str= "Where do cats enjoy purring?"):
        assert data_type in ['image', 'audio', 'text', 'qa', 'text_image'], "data_type needs to be image, audio, text, qa or text_image"
        assert model_type in ['bi_encoder', 'encoder'], "model_type needs to be bi_encoder or encoder"
        self.model = model
        self.vector_length = vector_length
        self.model_type = model_type
        self.data_type = data_type
        self.image_url = image_url
        self.audio_url = audio_url
        self.audio_sample_rate =  16000
        self.sentence = sample_sentence
        self.question = sample_question
    
    def assert_encode_works(self):
        if self.data_type == 'image':
            assert_vector_works(self.model.encode(self.image_url), self.vector_length)
        elif self.data_type == 'audio':
            assert_vector_works(self.model.encode(self.audio_url), self.vector_length)
        elif self.data_type == 'text':
            assert_vector_works(self.model.encode(self.sentence), self.vector_length)
        elif self.data_type == 'qa':
            assert_vector_works(self.model.encode_question(self.question), self.vector_length)
            assert_vector_works(self.model.encode_answer(self.sentence), self.vector_length)
        elif self.data_type == 'text_image':
            assert_vector_works(self.model.encode_text(self.question), self.vector_length)
            assert_vector_works(self.model.encode_image(self.image_url), self.vector_length)

    def assert_bulk_encode_works(self):
        if self.data_type == 'image':
            assert_vector_works(self.model.bulk_encode([self.image_url, self.image_url, self.image_url]), self.vector_length)
        elif self.data_type == 'audio':
            assert_vector_works(self.model.bulk_encode([self.audio_url, self.audio_url, self.audio_url]), self.vector_length)
        elif self.data_type == 'text':
            assert_vector_works(self.model.bulk_encode([self.sentence, self.sentence, self.sentence]), self.vector_length)
        elif self.data_type == 'qa':
            assert_vector_works(self.model.encode_answer(self.sentence), self.vector_length)
        elif self.data_type == 'text_image':
            assert_vector_works(self.model.encode_image(self.image_url), self.vector_length)

    def assert_encoding_methods_work(self):
        if self.model_type == 'encoder':
            self.assert_encode_works()
            self.assert_bulk_encode_works()
        elif self.model_type == 'bi_encoder':
            self.assert_biencode_works()
            self.assert_bulk_biencode_works()
    
    def assert_biencode_works(self):
        if self.data_type == 'qa':
            assert_vector_works(self.model.encode_question(self.question), self.vector_length)
            assert_vector_works(self.model.encode_answer(self.sentence), self.vector_length)
        elif self.data_type == 'text_image':
            assert_vector_works(self.model.encode_text(self.sentence), self.vector_length)
            assert_vector_works(self.model.encode_image(self.image_url), self.vector_length)

    def assert_bulk_biencode_works(self):
        if self.data_type == 'text':
            assert_vector_works(self.model.encode_answer(self.sentence), self.vector_length)
    
    @property
    def sample_document(self):
        """Sample documents.
        """
        return {
            'image_url': 'https://getvectorai.com/_nuxt/img/dog-1.3cc5fe1.png',
            'audio_url': 'https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_2.wav',
            'text': "Cats love purring on the beach.",
            'question': "Where do cats love purring?"
        }
    
    @property
    def sample_documents(self):
        return [self.sample_document] * 30

    @property
    def field_to_encode_mapping(self):
        if self.data_type == 'text':
            return 'text'
        if self.data_type == 'image':
            return 'image_url'
        if self.data_type == 'audio':
            return 'audio_url'
        if self.data_type == 'qa':
            return 'question'
        if self.data_type == 'text_image':
            return 'image_url'

    @property
    def field_to_search_mapping(self):
        if self.data_type == 'text':
            return 'text'
        if self.data_type == 'image':
            return 'image_url'
        if self.data_type == 'audio':
            return 'audio_url'
        if self.data_type == 'qa':
            return 'question'
        if self.data_type == 'text_image':
            return 'question'

    @property
    def random_string(self, length=8):
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(length))

    @property
    def vi_client(self):
        if 'VH_USERNAME' in os.environ.keys():
            return ViClient(os.environ['VH_USERNAME'], os.environ['VH_API_KEY'])
        elif 'VI_USERNAME' in os.environ.keys():
            return ViClient(os.environ['VI_USERNAME'], os.environ['VI_API_KEY'])
        return ViClient()

    def assert_insert_vectorai_simple(self):
        CN = 'test_vectorhub_' + self.random_string
        with TempClient(self.vi_client, CN) as client:
            response = client.insert_documents(CN, self.sample_documents,
            {self.field_to_encode_mapping: self.model})
            assert len(response['failed_document_ids']) == 0

    def assert_insert_vectorai_bulk_encode(self):
        CN = 'test_vectorhub_' + self.random_string
        with TempClient(self.vi_client, CN) as client:
            if self.model_type == 'encoder':
                response = client.insert_documents(CN,
                self.sample_documents,
                {self.field_to_encode_mapping: self.model},
                use_bulk_encode=True)
                assert len(response['failed_document_ids']) == 0
            elif self.model_type =='bi_encoder':
                response = client.insert_documents(CN, 
                self.sample_documents,
                {self.field_to_encode_mapping: self.model},
                use_bulk_encode=True)

    def assert_insert_vectorai_with_multiprocessing(self):
        CN = 'test_vectorhub_' + self.random_string
        with TempClient(self.vi_client, CN) as client:
            response = client.insert_documents(CN,
            self.sample_documents,
            {self.field_to_encode_mapping: self.model},
            use_bulk_encode=False, workers=4)
            assert len(response['failed_document_ids']) == 0

    def assert_insert_vectorai_with_multiprocessing_with_bulk_encode(self):
        CN = 'test_vectorhub_' + self.random_string
        with TempClient(self.vi_client, CN) as client:
            response = client.insert_documents(
                CN,
                self.sample_documents,
                {self.field_to_encode_mapping: self.model},
                use_bulk_encode=True, workers=4)
            assert len(response['failed_document_ids']) == 0

    def assert_simple_insertion_works(self):
        # Ensure that inserting in a collection works normally
        cn = 'test_vectorhub_' + self.random_string 
        items = self.vi_client.get_field_across_documents(
            self.field_to_encode_mapping, self.sample_documents
        )
        self.model.add_documents(self.vi_client.username, self.vi_client.api_key, items, collection_name=cn)
        time.sleep(2)
        response = self.model.search(self.sample_document[self.field_to_search_mapping])
        self.vi_client.delete_collection(cn)
        assert len(response['results']) > 0


    def assert_insertion_into_vectorai_works(self):
        self.assert_simple_insertion_works()
        self.assert_insert_vectorai_simple()
        self.assert_insert_vectorai_bulk_encode()
        # Remove tests for now due to local object pickling
        # self.assert_insert_vectorai_with_multiprocessing()
        # self.assert_insert_vectorai_with_multiprocessing_with_bulk_encode()


def assert_encoder_works(model, vector_length=None, data_type='image', model_type='encoder',
    image_url: str='https://getvectorai.com/_nuxt/img/dog-1.3cc5fe1.png',
    audio_url: str='https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_2.wav',
    sample_sentence: str= "Cats enjoy purring in the nature.",
    sample_question: str= "Where do cats enjoy purring?"):
    """
    Assert that an encoder works
    """
    if vector_length is None:
        try:
            # Use the embedded URL module for now. 
            vector_length = model.urls[model.model_url]['vector_length']
        except:
            pass
    model_check = AssertModelWorks(model=model, vector_length=vector_length, 
    data_type=data_type, model_type=model_type, image_url=image_url,audio_url=audio_url, 
    sample_sentence=sample_sentence, sample_question=sample_question)
    model_check.assert_encoding_methods_work()
    model_check.assert_insertion_into_vectorai_works()
    gc.collect()
