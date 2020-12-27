import numpy as np
from vectorhub.utils import list_models, list_installed_models

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
    if isinstance(vector[0], list):
        # If this is a list of vectors as opposed to just one
        for v in vector:
            assert not is_dummy_vector(vector[0], vector_length),  "Is a dummy vector"
            if vector_length is not None:
                assert len(vector[0]) == vector_length, f"Does not match vector length of {vector_length}"
    else:
        assert not is_dummy_vector(vector, vector_length),  "Is a dummy vector"
        if vector_length is not None:
            assert len(vector) == vector_length, f"Does not match vector length of {vector_length}"

class AssertModelWorks:
    def __init__(self, model, vector_length, model_type='image'):
        assert model_type in ['image', 'audio', 'text'], "Needs to be image, audio or text"
        self.model = model
        self.vector_length = vector_length
        self.model_type = model_type
        self.image_url = 'https://getvectorai.com/assets/logo-square.png'
        self.audio_url = 'https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_2.wav'
        self.audio_sample_rate =  16000
        self.sentence = "Cats enjoy purring in the nature."
    
    def assert_encode_works(self):
        if self.model_type == 'image':
            assert_vector_works(self.model.encode(self.image_url), self.vector_length)
        elif self.model_type == 'audio':
            assert_vector_works(self.model.encode(self.audio_url), self.vector_length)
        elif model_type == 'text':
            assert_vector_works(self.model.encode(self.sentence), self.vector_length)

    def assert_bulk_encode_works(self):
        if self.model_type == 'image':
            assert_vector_works(self.model.bulk_encode([self.image_url, self.image_url, self.image_url]), self.vector_length)
        elif self.model_type == 'audio':
            assert_vector_works(self.model.bulk_encode([self.audio_url, self.audio_url, self.audio_url]), self.vector_length)
        elif model_type == 'text':
            assert_vector_works(self.model.bulk_encode([self.sentence, self.sentence, self.sentence]), self.vector_length)

    def assert_encoding_methods_work(self):
        self.assert_encode_works()
        self.assert_bulk_encode_works()

def assert_model_works(model, vector_length, model_type='image'):
    """
    Assert that a model works.
    """
    assert_model = AssertModelWorks(model=model, vector_length=vector_length, model_type=model_type)
    assert_model.assert_encoding_methods_work()
