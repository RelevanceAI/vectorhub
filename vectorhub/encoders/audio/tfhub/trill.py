from ....import_utils import *
from ....models_dict import *
if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-audio-tfhub-trill']):
    import tensorflow as tf
    import tensorflow_hub as hub
    import traceback
from ..base import BaseAudio2Vec
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition

TrillModelDefinition = ModelDefinition(
    model_name="Trill - Triplet Loss Network", 
    model_id = "audio/trill",
    vector_length=512, 
    description="""
    The ultimate goal of transfer learning is to reduce labeled data requirements by exploiting a pre-existing embedding model trained for 
    different datasets or tasks. The visual and language communities have established benchmarks to compare embeddings, but the speech 
    community has yet to do so. This paper proposes a benchmark for comparing speech representations on non-semantic tasks, and proposes a 
    representation based on an unsupervised triplet-loss objective. The proposed representation outperforms other representations on the 
    benchmark, and even exceeds state-of-the-art performance on a number of transfer learning tasks. The embedding is trained on a publicly 
    available dataset, and it is tested on a variety of low-resource downstream tasks, including personalization tasks and medical domain. 
    The benchmark, models, and evaluation code are publicly released.""",
    paper="https://arxiv.org/abs/2002.12764",
    repo="https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/3",
    installation="pip install vectorhub[{}]".format(MODEL_REQUIREMENTS['encoders-audio-tfhub-trill']),
    example="""
    #pip install vectorhub[{}]
    from vectorhub.encoders.audio.tfhub import Trill2Vec
    model = Trill2Vec()
    sample = model.read('https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_2.wav')
    model.encode(sample)
    """.format(MODEL_REQUIREMENTS['encoders-audio-tfhub-trill'])
)

__doc__ = TrillModelDefinition.create_docs()


class Trill2Vec(BaseAudio2Vec):
    definition = TrillModelDefinition
    def __init__(self, model_url: str = 'https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/3', 
    layer: str = 'embedding'):
        self.model_url = model_url
        self.layer = layer
        self.model = hub.load(self.model_url)
        self.model_name = model_url.replace(
            'https://tfhub.dev/google/', '').replace('/', '_')
        self.vector_length = 512
    
    @catch_vector_errors
    def encode(self, audio, vector_operation='mean'):
        """
        
        Example:
            >>> from encoders.audio.trill import Trill2Vec
            >>> encoder = Trill2Vec()
            >>> encoder.encode(...)
        
        """
        return self._vector_operation(self.model(samples=audio, sample_rate=16000)[self.layer], vector_operation)

    @catch_vector_errors
    def bulk_encode(self, data, vector_operation='mean'):
        return [self.encode(c) for c in data]


TrillDistilledModelDefinition = ModelDefinition(
    model_name="Trill Distilled - Triplet Loss Network", 
    model_id = "audio/trill-distilled",
    vector_length=512, 
    description="""
    The ultimate goal of transfer learning is to reduce labeled data requirements by exploiting a pre-existing embedding model trained for 
    different datasets or tasks. The visual and language communities have established benchmarks to compare embeddings, but the speech 
    community has yet to do so. This paper proposes a benchmark for comparing speech representations on non-semantic tasks, and proposes a 
    representation based on an unsupervised triplet-loss objective. The proposed representation outperforms other representations on the 
    benchmark, and even exceeds state-of-the-art performance on a number of transfer learning tasks. The embedding is trained on a publicly 
    available dataset, and it is tested on a variety of low-resource downstream tasks, including personalization tasks and medical domain. 
    The benchmark, models, and evaluation code are publicly released.""",
    paper="https://arxiv.org/abs/2002.12764",
    repo="https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/3",
    installation="pip install vectorhub[encoders-audio-tfhub]",
    example="""
    #pip install vectorhub[encoders-audio-tfhub]
    from vectorhub.encoders.audio.tfhub import TrillDistilled2Vec
    model = TrillDistilled2Vec()
    sample = model.read('https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_2.wav')
    model.encode(sample)
    """
)


class TrillDistilled2Vec(BaseAudio2Vec):
    definition = TrillDistilledModelDefinition
    def __init__(self, model_url: str = 'https://tfhub.dev/google/nonsemantic-speech-benchmark/trill-distilled/3', layer: str = 'embedding'):
        self.model_url = model_url
        self.layer = layer
        self.model = hub.load(self.model_url)
        self.model_name = model_url.replace(
            'https://tfhub.dev/google/', '').replace('/', '_')
        self.vector_length = 2048

    @catch_vector_errors
    def encode(self, audio, vector_operation='mean'):
        return self._vector_operation(self.model(samples=audio, sample_rate=16000)[self.layer], vector_operation)