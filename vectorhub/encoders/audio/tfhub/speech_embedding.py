from datetime import date
from ....import_utils import *
from ....models_dict import MODEL_REQUIREMENTS
if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-audio-tfhub-speech_embedding']):
    import tensorflow as tf
    import tensorflow_hub as hub
    import traceback

from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ..base import BaseAudio2Vec

SpeechEmbeddingModelDefinition = ModelDefinition(markdown_filepath="encoders/audio/tfhub/speech_embedding.md")

__doc__ = SpeechEmbeddingModelDefinition.create_docs()

class SpeechEmbedding2Vec(BaseAudio2Vec):
    definition = SpeechEmbeddingModelDefinition
    def __init__(self, model_url: str = 'https://tfhub.dev/google/speech_embedding/1', signature: str = 'default'):
        self.model_url = model_url
        self.signature = signature
        self.model = hub.load(self.model_url).signatures[self.signature]
        self.model_name = self.model_url.replace(
            'https://tfhub.dev/google/', '').replace('/', '_')
        self.vector_length = 96

    @property
    def urls(self):
        return {
            'https://tfhub.dev/google/speech_embedding/1': {'vector_length': 96}
        }

    @catch_vector_errors
    def encode(self, audio, vector_operation='mean'):
        """
        Encode the vector.
        Example:

        >>> from vectorhub.encoders.audio import SpeechEmbedding2Vec
        >>> encoder = SpeechEmbedding2Vec()
        >>> encoder.encode(...)
        """
        if isinstance(audio, str):
            audio = self.read(audio)
        return self._vector_operation(self.model(tf.constant([audio]))[self.signature][0], vector_operation=vector_operation)[0]

    @catch_vector_errors
    def bulk_encode(self, audios, vector_operation='mean'):
        # TODO: Change list comprehension to tensor.
        # audios = [self.read(audio) if isinstance(audio, str) else audio for audio in audios]	        
        # return self._vector_operation(self.model(tf.constant(audios))[self.signature][0], vector_operation=vector_operation)
        # TODO: Change list comprehension to tensor.
        return [self.encode(x, vector_operation=vector_operation) for x in audios]
