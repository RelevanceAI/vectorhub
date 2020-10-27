from ....import_utils import *
from ....models_dict import MODEL_REQUIREMENTS
if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-audio-tfhub-vggish']):
    import tensorflow as tf
    import tensorflow_hub as hub

from ..base import BaseAudio2Vec
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition

VggishModelDefinition = ModelDefinition(
    model_id = "audio/vggish",
    model_name="VGGish", 
    vector_length=512, 
    description="""An audio event embedding model trained on the YouTube-8M dataset.
VGGish should be used:
- as a high-level feature extractor: the 128-D embedding output of VGGish can be used as the input features of another shallow model which can then be trained on a small amount of data for a particular task. This allows quickly creating specialized audio classifiers without requiring a lot of labeled data and without having to train a large model end-to-end.
- as a warm start: the VGGish model parameters can be used to initialize part of a larger model which allows faster fine-tuning and model exploration.
    """,
    limitations="""
    VGGish has been trained on millions of YouTube videos and although these are very diverse, there can still be a domain 
    mismatch between the average YouTube video and the audio inputs expected for any given task. You should expect to do some 
    amount of fine-tuning and calibration to make VGGish usable in any system that you build.
    """,
    repo="https://tfhub.dev/google/vggish/1",
    installation="pip install vectorhub[encoders-audio-tfhub]",
    example="""
    #pip install vectorhub[encoders-audio-tfhub]
    from vectorhub.encoders.audio.tfhub import Vggish2Vec
    model = Vggish2Vec()
    sample = model.read('https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_2.wav')
    model.encode(sample)
    """
)

__doc__ = VggishModelDefinition.create_docs()

class Vggish2Vec(BaseAudio2Vec):
    definition = VggishModelDefinition
    def __init__(self, model_url: str = 'https://tfhub.dev/google/vggish/1'):
        self.model_url = model_url
        self.model_name = self.model_url.replace(
            'https://tfhub.dev/google/', '').replace('/', '')
        self.model = hub.load(self.model_url)
        self.vector_length = 128

    @catch_vector_errors
    def encode(self, audio, vector_operation='mean'):
        return self._vector_operation(self.model(audio), vector_operation)