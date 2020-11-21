from vectorhub.auto_encoder import AutoEncoder, ENCODER_MAPPINGS, list_all_auto_models, BIENCODER_MAPPINGS, AutoBiEncoder
import warnings
import pytest

def is_vector_default(vector):
    """
        Return True if the vector is the default vector, False if it is not.
    """
    return vector[0] == 1e-7 and vector[1] == 1e-7 and vector[2] == 1e-7

@pytest.mark.parametrize('name', list(ENCODER_MAPPINGS.keys()))
def test_encoders_instantiation(name):
    encoder = AutoEncoder.from_model(name)
    if name not in ['text/use-lite']:
        if 'text' in name:
            result = encoder.encode("HI")
            # We set this because sometimes it may 
            # result in 2 separate vectors.
        if 'image' in name:
            sample = encoder.read('https://getvectorai.com/assets/logo-square.png')
            result = encoder.encode(sample)
            assert is_vector_default(result)
            sample = encoder.to_grayscale(encoder.read('https://getvectorai.com/assets/logo-square.png'))
            result = encoder.encode(sample)
            assert is_vector_default(result)
        if 'audio' in name:
            sample = encoder.read(
            'https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_2.wav', 16000
            )
            result = encoder.encode(sample)
        # Check to ensure that this isn't just the default vector
        assert not is_vector_default(result)
        assert len(result) > 10

@pytest.mark.parametrize('name', list(BIENCODER_MAPPINGS.keys()))
def test_biencoder_mappings(name):
    bi_encoder = AutoBiEncoder.from_model(name)
    if 'text_text' in name:
        vector = bi_encoder.encode_question("Why?")
        assert len(vector) > 10
        vector = bi_encoder.encode_answer("Yes!")
        assert len(vector) > 10
    assert True

def test_listing_all_models():
    """
        Simple test to ensure model listing works.
    """
    assert len(list_all_auto_models()) > 1
