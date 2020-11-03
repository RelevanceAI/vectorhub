from ..base import BaseAudio2Vec
from ....base import catch_vector_errors 
from ....doc_utils import ModelDefinition
from ....import_utils import *
from ....models_dict import MODEL_REQUIREMENTS
from datetime import date
if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-audio-pytorch-fairseq']):
    import torch
    from fairseq.models.wav2vec import Wav2VecModel
    import numpy as np

WavModelDefinition = ModelDefinition(
    model_id="audio/wav2vec",
    model_name="Wav2Vec", 
    vector_length=512,
    description="""We show for the first time that learning powerful representations from speech audio alone followed by fine-tuning on transcribed speech can outperform the best semi-supervised methods while being conceptually simpler. wav2vec 2.0 masks the speech input in the latent space and solves a contrastive task defined over a quantization of the latent representations which are jointly learned. Experiments using all labeled data of Librispeech achieve 1.8/3.3 WER on the clean/noisy test sets. When lowering the amount of labeled data to one hour, wav2vec 2.0 outperforms the previous state of the art on the 100 hour subset while using 100 times less labeled data. Using just ten minutes of labeled data and pre-training on 53k hours of unlabeled data still achieves 5.2/8.6 WER on the noisy/clean test sets of Librispeech. This demonstrates the feasibility of speech recognition with limited amounts of labeled data..""",
    paper="https://arxiv.org/abs/2006.11477",
    repo="https://github.com/pytorch/fairseq",
    installation="pip install vectorhub[encoders-audio-pytorch]",
    release_date=date(2020,6,20),
    example="""
    #pip install vectorhub[encoders-audio-pytorch]
    from vectorhub.encoders.audio.pytorch import Wav2Vec
    model = Wav2Vec()
    sample = model.read('https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_2.wav')
    model.encode(sample)
    """
)

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

    @catch_vector_errors
    def encode(self, audio, vector_operation='mean'):
        """
        Example:

            >>> from vectorhub.encoders.audio import Wav2Vec
            >>> encoder = Wav2Vec()
            >>> encoder.encode("...")
        """
        return self._vector_operation(self.model.feature_extractor(torch.from_numpy(np.array([audio]))).detach().numpy().tolist()[0], vector_operation=vector_operation, axis=1)
