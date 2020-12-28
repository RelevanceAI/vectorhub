from datetime import date
from ....base import catch_vector_errors 
from ....doc_utils import ModelDefinition
from ....import_utils import *
from ....models_dict import MODEL_REQUIREMENTS
from ..base import BaseAudio2Vec

if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-audio-pytorch-fairseq']):
    import torch
    import numpy as np
    from fairseq.models.wav2vec import Wav2VecModel

WavModelDefinition = ModelDefinition(markdown_filepath='encoders/audio/pytorch/wav2vec')

class Wav2Vec(BaseAudio2Vec):
    definition = WavModelDefinition
    def __init__(self, model_url: str = 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt'):
        self.list_of_urls = [
            'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt',
            'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small_10m.pt',
            'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small_100h.pt',
            'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small_960h.pt',
            'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox.pt',
            'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_10m.pt',
            'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_100h.pt',
            'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec2_vox_960h.pt',
        ]
        self.validate_model_url(model_url, self.list_of_urls)
        self.init(model_url)
        self.vector_length = 512

    def init(self, model_url: str):
        self.model_url = model_url
        self.model_name = self.model_url.replace(
            'https://dl.fbaipublicfiles.com/fairseq/', '').replace('/', '_')
        torch_model = torch.hub.load_state_dict_from_url(self.model_url)
        self.model = Wav2VecModel.build_model(torch_model['args'], task=None)

    @property
    def urls(self):
        return {
            'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt': {},
            'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small_10m.pt': {},
            'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small_100h.pt': {},
            'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small_960h.pt': {}, 
            'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox.pt': {},
            'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_10m.pt': {},
            'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_100h.pt': {},
            'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec2_vox_960h.pt': {},
        }

    @catch_vector_errors
    def encode(self, audio, vector_operation='mean'):
        """
        Example:

            >>> from vectorhub.encoders.audio import Wav2Vec
            >>> encoder = Wav2Vec()
            >>> encoder.encode("...")
        """
        if isinstance(audio, str):
            audio = self.read(audio)
        return self._vector_operation(self.model.feature_extractor(torch.from_numpy(np.array([audio]))).detach().numpy().tolist()[0], vector_operation=vector_operation, axis=1)

    @catch_vector_errors
    def bulk_encode(self, audios, vector_operation='mean'):
        return [self.encode(audio, vector_operation=vector_operation) for audio in audios]
