import warnings
import pytest
from vectorhub.auto_encoder import AutoEncoder, ENCODER_MAPPINGS, list_all_auto_models, BIENCODER_MAPPINGS, AutoBiEncoder
from .test_utils import *

@pytest.mark.audio
@pytest.mark.parametrize('name', list(ENCODER_MAPPINGS.keys()))
def test_encoders_instantiation_audio(name):
    if 'audio' in name:
        encoder = AutoEncoder.from_model(name)
        assert_encoder_works(encoder, model_type='audio')
    else:
        # Default to test passing otherwise
        assert True

@pytest.mark.text
@pytest.mark.parametrize('name', list(ENCODER_MAPPINGS.keys()))
def test_encoders_instantiation_text(name):
    if name not in ['text/use-lite', 'text/elmo']:
        if 'text' in name:
            encoder = AutoEncoder.from_model(name)
            assert_encoder_works(encoder, model_type='text')
        else:
            # Default to test passing otherwise
            assert True


@pytest.mark.image
@pytest.mark.parametrize('name', list(ENCODER_MAPPINGS.keys()))
def test_encoders_instantiation_image(name):
    if 'image' in name:
        encoder = AutoEncoder.from_model(name)
        assert_encoder_works(encoder)
        if 'fastai' not in name:
            sample = encoder.to_grayscale(encoder.read('https://getvectorai.com/assets/logo-square.png'))
            result = encoder.encode(sample)
            assert not is_dummy_vector(result)
    else:
        # Default to test passing otherwise
        assert True

@pytest.mark.text
@pytest.mark.parametrize('name', list(BIENCODER_MAPPINGS.keys()))
def test_auto_biencoders(name):
    if 'text_text' in name:
        bi_encoder = AutoBiEncoder.from_model(name)
        assert_encoder_works(bi_encoder)
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
