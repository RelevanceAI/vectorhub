"""
An audio URL.
"""
import pytest 
import os
from vectorai import ViClient

def audio_url():
    return 'https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_2.wav'


@pytest.fixture
def dummy_username():
    if 'VI_USERNAME' in os.environ.keys():
        return os.environ['VI_USERNAME']
    elif 'VH_USERNAME' in os.environ.keys():
        return os.environ['VH_USERNAME']

@pytest.fixture
def dummy_api_key():
    if 'VI_USERNAME' in os.environ.keys():
        return os.environ['VI_API_KEY']
    elif 'VH_USERNAME' in os.environ.keys():
        return os.environ['VH_API_KEY']

@pytest.fixture
def dummy_client(dummy_username, dummy_api_key):
    return ViClient(dummy_username, dummy_api_key)
