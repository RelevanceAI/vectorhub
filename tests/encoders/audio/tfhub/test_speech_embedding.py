from vectorhub.encoders.audio.tfhub.speech_embedding import SpeechEmbedding2Vec
from ....test_utils import assert_encoder_works

def test_speech_embedding_works():
    """
    Testing for speech embedding initialization
    """
    encoder = SpeechEmbedding2Vec()
    assert_encoder_works(encoder, vector_length=96, data_type='audio')
