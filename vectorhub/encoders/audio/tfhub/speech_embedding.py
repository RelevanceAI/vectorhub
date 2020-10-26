from ....import_utils import *
from ....models_dict import MODEL_REQUIREMENTS
if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-audio-tfhub-speech_embedding']):
    import tensorflow as tf
    import tensorflow_hub as hub
    import traceback

from ..base import BaseAudio2Vec
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition

SpeechEmbeddingModelDefinition = ModelDefinition(
    model_id = "audio/speech_embedding",
    model_name="Speech Embedding", 
    vector_length=96,
    description="""
With the rise of low power speech-enabled devices, there is a growing demand to quickly produce models for recognizing arbitrary 
sets of keywords. As with many machine learning tasks, one of the most challenging parts in the model creation process is obtaining
a sufficient amount of training data. In this paper, we explore the effectiveness of synthesized speech data in training small, 
spoken term detection models of around 400k parameters. Instead of training such models directly on the audio or low level features
such as MFCCs, we use a pre-trained speech embedding model trained to extract useful features for keyword spotting models. Using this 
speech embedding, we show that a model which detects 10 keywords when trained on only synthetic speech is equivalent to a model trained 
on over 500 real examples. We also show that a model without our speech embeddings would need to be trained on over 4000 real examples to 
reach the same accuracy.""",
    paper="https://arxiv.org/abs/2002.01322",
    repo="https://tfhub.dev/google/speech_embedding/1",
    installation="pip install vectorhub[encoders-audio-tfhub]",
    example="""
    #pip install vectorhub[encoders-audio-tfhub]
    from vectorhub.encoders.audio.tfhub import SpeechEmbedding2Vec
    model = SpeechEmbedding2Vec()
    sample = model.read('https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_2.wav')
    model.encode(sample)
    """
)

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

    @catch_vector_errors
    def encode(self, audio, vector_operation='mean'):
        """
        Encode the vector.
        Example:

        >>> from vectorhub.encoders.audio import SpeechEmbedding2Vec
        >>> encoder = SpeechEmbedding2Vec()
        >>> encoder.encode(...)
        """
        return self._vector_operation(self.model(tf.constant([audio]))[self.signature][0], vector_operation=vector_operation)[0]

    @catch_vector_errors
    def bulk_encode(self, audios, vector_operation='mean'):
        return self._vector_operation(self.model(tf.constant(audios))[self.signature][0], vector_operation=vector_operation)
