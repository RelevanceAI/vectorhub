from vectorhub.encoders.audio.tfhub.speech_embedding import SpeechEmbedding2Vec


def test_speech_embedding_initialize():
    """
    Testing for speech embedding initialization
    """
    client = SpeechEmbedding2Vec()
    assert True


def test_speech_embedding_encode():
    """
    Testing for speech embedding single encode
    """
    client = SpeechEmbedding2Vec()
    audio = client.read(
        'https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_2.wav', 
    16000)
    embedding = client.encode(audio)
    assert embedding != None
    assert len(embedding) == 96

def test_speech_embedding_bulk_encode():
    """
    Testing for speech embedding multi encode
    """
    client = SpeechEmbedding2Vec()
    list_of_audio = [
        'https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_2.wav',
        'https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_2.wav'
    ]
    audio = [client.read(c) for c in list_of_audio]
    embeddings = client.bulk_encode(audio)
    assert True
