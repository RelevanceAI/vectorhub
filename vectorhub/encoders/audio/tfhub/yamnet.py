from ....import_utils import *
from ....models_dict import MODEL_REQUIREMENTS
if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-audio-tfhub-yamnet']):
    import tensorflow as tf
    import tensorflow_hub as hub

from ..base import BaseAudio2Vec
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from datetime import date

YamnetModelDefinition = ModelDefinition(
    model_id="audio/yamnet",
    model_name="Yamnet", 
    vector_length=1024,
    description="""
    YAMNet is an audio event classifier that takes audio waveform as input and makes independent predictions for each 
    of 521 audio events from the AudioSet ontology. The model uses the MobileNet v1 architecture and was trained using 
    the AudioSet corpus. This model was originally released in the TensorFlow Model Garden, where we have the model 
    source code, the original model checkpoint, and more detailed documentation.
    This model can be used: 
    
    - as a stand-alone audio event classifier that provides a reasonable baseline across a wide variety of audio events.
    - as a high-level feature extractor: the 1024-D embedding output of YAMNet can be used as the input features of another shallow model which can then be trained on a small amount of data for a particular task. This allows quickly creating specialized audio classifiers without requiring a lot of labeled data and without having to train a large model end-to-end.
    - as a warm start: the YAMNet model parameters can be used to initialize part of a larger model which allows faster fine-tuning and model exploration.
    """,
    release_date=date(2020,3,11),
    limitations="""
    YAMNet's classifier outputs have not been calibrated across classes, so you cannot directly treat 
    the outputs as probabilities. For any given task, you will very likely need to perform a calibration with task-specific data 
    which lets you assign proper per-class score thresholds and scaling.
    YAMNet has been trained on millions of YouTube videos and although these are very diverse, there can still be a domain mismatch 
    between the average YouTube video and the audio inputs expected for any given task. You should expect to do some amount of 
    fine-tuning and calibration to make YAMNet usable in any system that you build.""",
    repo="https://tfhub.dev/google/yamnet/1",
    installation="pip install vectorhub[encoders-audio-tfhub]",
    example="""
    #pip install vectorhub[encoders-audio-tfhub]
    from vectorhub.encoders.audio.tfhub import Yamnet2Vec
    model = Yamnet2Vec()
    sample = model.read('https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_2.wav')
    model.encode(sample)
    """
)

__doc__ = YamnetModelDefinition.create_docs()

class Yamnet2Vec(BaseAudio2Vec):
    definition = YamnetModelDefinition
    def __init__(self, model_url: str = 'https://tfhub.dev/google/yamnet/1'):
        self.model_url = model_url
        self.model = hub.load(self.model_url)
        self.model_name = self.model_url.replace(
            'https://tfhub.dev/google/', '').replace('/', '_')
        self.vector_length = 1024

    @catch_vector_errors
    def encode(self, audio, vector_operation='mean', layer='embeddings'):
        outputs = self.model(audio)
        if layer == 'scores':
            return self._vector_operation(outputs[0], vector_operation)
        elif layer == 'log_mel_spectrogram':
            return self._vector_operation(outputs[2], vector_operation)
        else:
            return self._vector_operation(outputs[1], vector_operation)