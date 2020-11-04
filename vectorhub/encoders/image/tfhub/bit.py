from ..base import BaseImage2Vec
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import *
if is_all_dependency_installed('encoders-image-tfhub'):
    import tensorflow as tf
    import tensorflow_hub as hub
    import traceback
from datetime import date

BITModelDefinition = ModelDefinition(
    model_id = "image/bit",
    model_name="BiT - Big Transfer, General Visual Representation Learning (Small)", 
    vector_length=2048, 
    description="""Transfer of pre-trained representations improves sample efficiency and simplifies hyperparameter tuning when training 
    deep neural networks for vision. We revisit the paradigm of pre-training on large supervised datasets and fine-tuning the model 
    on a target task. We scale up pre-training, and propose a simple recipe that we call Big Transfer (BiT). By combining a few carefully 
    selected components, and transferring using a simple heuristic, we achieve strong performance on over 20 datasets. BiT performs well across 
    a surprisingly wide range of data regimes -- from 1 example per class to 1M total examples. BiT achieves 87.5% top-1 accuracy on ILSVRC-2012, 
    99.4% on CIFAR-10, and 76.3% on the 19 task Visual Task Adaptation Benchmark (VTAB). On small datasets, BiT attains 76.8% on 
    ILSVRC-2012 with 10 examples per class, and 97.0% on CIFAR-10 with 10 examples per class. We conduct detailed analysis 
    of the main components that lead to high transfer performance.""",
    paper="https://arxiv.org/abs/1912.11370", 
    repo="https://github.com/google-research/big_transfer",
    installation="pip install vectorhub[encoders-image-tfhub]",
    release_date=date(2019,12,24),
    example="""
    #pip install vectorhub[encoders-image-tfhub]
    from vectorhub.encoders.image.tfhub import BitSmall2Vec
    model = BitSmall2Vec()
    sample = model.read('https://getvectorai.com/assets/hub-logo-with-text.png')
    model.encode(sample)
    """
)

__doc__ = BITModelDefinition.create_docs()

class BitSmall2Vec(BaseImage2Vec):
    definition = BITModelDefinition
    def __init__(self, model_url: str = "https://tfhub.dev/google/bit/s-r50x1/1"):
        list_of_urls = {
            'https://tfhub.dev/google/bit/s-r50x1/1': {"vector_length":2048}, # 2048 output shape
            'https://tfhub.dev/google/bit/s-r50x3/1': {"vector_length":6144},   # 6144 output shape
            'https://tfhub.dev/google/bit/s-r101x1/1': {"vector_length":2048},  # 2048 output shape
            'https://tfhub.dev/google/bit/s-r101x3/1': {"vector_length":6144},  # 6144 output shape
            'https://tfhub.dev/google/bit/s-r152x4/1': {"vector_length":8192},   # 8192 output shape
        }
        self.validate_model_url(model_url, list_of_urls)
        self.init(model_url)
        self.vector_length = list_of_urls[model_url]["vector_length"]

    def init(self, model_url: str):
        self.model_url = model_url
        self.model_name = self.model_url.replace(
            'https://tfhub.dev/google/', '').replace('/', '_')
        self.model = hub.load(self.model_url)

    @catch_vector_errors
    def encode(self, image):
        return self.model([image]).numpy().tolist()[0]

    @catch_vector_errors
    def bulk_encode(self, images, threads=10, chunks=10):
        return [i for c in self.chunk(images, chunks) for i in self.model(c).numpy().tolist()]


BITMediumModelDefinition = ModelDefinition(
    model_id = "image/bit-medium",
    model_name="BiT Medium - Big Transfer, General Visual Representation Learning (Medium)", 
    vector_length=2048, 
    description="""Transfer of pre-trained representations improves sample efficiency and simplifies hyperparameter tuning when training 
    deep neural networks for vision. We revisit the paradigm of pre-training on large supervised datasets and fine-tuning the model 
    on a target task. We scale up pre-training, and propose a simple recipe that we call Big Transfer (BiT). By combining a few carefully 
    selected components, and transferring using a simple heuristic, we achieve strong performance on over 20 datasets. BiT performs well across 
    a surprisingly wide range of data regimes -- from 1 example per class to 1M total examples. BiT achieves 87.5% top-1 accuracy on ILSVRC-2012, 
    99.4% on CIFAR-10, and 76.3% on the 19 task Visual Task Adaptation Benchmark (VTAB). On small datasets, BiT attains 76.8% on 
    ILSVRC-2012 with 10 examples per class, and 97.0% on CIFAR-10 with 10 examples per class. We conduct detailed analysis 
    of the main components that lead to high transfer performance.""",
    paper="https://arxiv.org/abs/1912.11370", 
    repo="https://github.com/google-research/big_transfer",
    installation="pip install vectorhub[encoders-image-tfhub]",
    example="""
    #pip install vectorhub[encoders-image-tfhub]
    from vectorhub.encoders.image.tfhub import BitMedium2Vec
    model = BitMedium2Vec()
    sample = model.read('https://getvectorai.com/assets/hub-logo-with-text.png')
    model.encode(sample)
    """
)

class BitMedium2Vec(BitSmall2Vec):
    definition = BITMediumModelDefinition
    def __init__(self, model_url: str = 'https://tfhub.dev/google/bit/m-r50x1/1'):
        list_of_urls = {
            'https://tfhub.dev/google/bit/m-r50x1/1': {"vector_length":2048},  # 2048 output shape
            'https://tfhub.dev/google/bit/m-r50x3/1': {"vector_length":6144}, # 6144 output shape
            'https://tfhub.dev/google/bit/m-r101x1/1': {"vector_length":2048},  # 2048 output shape
            'https://tfhub.dev/google/bit/m-r101x3/1': {"vector_length":6144},  # 6144 output shape
            'https://tfhub.dev/google/bit/m-r152x4/1': {"vector_length":8192},  # 8192 output shape
        }
        self.validate_model_url(model_url, list_of_urls)
        self.init(model_url)
        self.vector_length = list_of_urls[model_url]["vector_length"]
