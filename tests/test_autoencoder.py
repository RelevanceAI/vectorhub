import warnings
import pytest
from vectorhub.auto_encoder import AutoEncoder, ENCODER_MAPPINGS, list_all_auto_models, BIENCODER_MAPPINGS, AutoBiEncoder
from .test_utils import *

@pytest.mark.audio
@pytest.mark.parametrize('name', list(ENCODER_MAPPINGS.keys())[0:3])
def test_encoders_instantiation_audio(name):
    if 'audio' in name:
        encoder = AutoEncoder.from_model(name)
        assert_encoder_works(encoder, data_type='audio')
    else:
        # Default to test passing otherwise
        assert True

@pytest.mark.text
@pytest.mark.parametrize('name', list(ENCODER_MAPPINGS.keys())[0:3])
def test_encoders_instantiation_text(name):
    if name not in ['text/use-lite', 'text/elmo']:
        if 'text' in name:
            encoder = AutoEncoder.from_model(name)
            assert_encoder_works(encoder, data_type='text')
        else:
            # Default to test passing otherwise
            assert True


@pytest.mark.image
@pytest.mark.parametrize('name', list(ENCODER_MAPPINGS.keys())[0:3])
def test_encoders_instantiation_image(name):
    if 'image' in name:
        encoder = AutoEncoder.from_model(name)
        assert_encoder_works(encoder, data_type='image')
        if 'fastai' not in name:
            sample = encoder.to_grayscale(encoder.read('https://getvectorai.com/_nuxt/img/dog-1.3cc5fe1.png'))
            result = encoder.encode(sample)
            assert not is_dummy_vector(result)
    else:
        # Default to test passing otherwise
        assert True

@pytest.mark.text
@pytest.mark.parametrize('name', list(BIENCODER_MAPPINGS.keys())[0:3])
def test_auto_biencoders(name):
    if 'qa' in name:
        bi_encoder = AutoBiEncoder.from_model(name)
        assert_encoder_works(bi_encoder, data_type='text', model_type='bi_encoder')

def test_listing_all_models():
    """
        Simple test to ensure model listing works.
    """
    assert len(list_all_auto_models()) > 1
