from ..base import BaseImage2Vec
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import *
from ....models_dict import MODEL_REQUIREMENTS
from datetime import date
if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-image-tfhub-resnet']):
    import tensorflow as tf
    import tensorflow_hub as hub

ResNetModelDefinition = ModelDefinition(
    model_id = "image/resnet",
    model_name="ResNet", 
    vector_length=2048, 
    description="""Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers---8x deeper than VGG nets but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers.""",
    paper="https://arxiv.org/abs/1512.03385",
    installation="pip install vectorhub[encoders-image-tfhub]",
    release_date=date(2015,12,10),
    example="""
    #pip install vectorhub[encoders-image-tfhub]
    from vectorhub.encoders.image.tfhub import ResnetV12Vec
    model = ResnetV12Vec()
    sample = model.read('https://getvectorai.com/assets/hub-logo-with-text.png')
    model.encode(sample)
    """
)

__doc__ = ResNetModelDefinition.create_docs()


class ResnetV12Vec(BaseImage2Vec):
    definition = ResNetModelDefinition
    def __init__(self, model_url: str = 'https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/4'):
        list_of_urls = [
            # 50 layers
            'https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/4',

            # 101 layers
            'https://tfhub.dev/google/imagenet/resnet_v1_101/feature_vector/4',

            # 152 layers
            'https://tfhub.dev/google/imagenet/resnet_v1_152/feature_vector/4',
        ]
        self.validate_model_url(model_url, list_of_urls)
        self.init(model_url)
        self.vector_length = 2048

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


ResNetV2ModelDefinition = ModelDefinition(
    model_id='image/resnet-v2',
    model_name="ResNet", 
    vector_length=2048, 
    description="""Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers---8x deeper than VGG nets but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers.""",
    paper="https://arxiv.org/abs/1512.03385",
    installation="pip install vectorhub['encoders-image-tfhub']",
    example="""
    #pip install vectorhub['encoders-image-tfhub']
    from vectorhub.encoders.image.tfhub import ResnetV22Vec
    model = ResnetV22Vec()
    sample = model.read('https://getvectorai.com/assets/hub-logo-with-text.png')
    model.encode(sample)
    """
)

class ResnetV22Vec(ResnetV12Vec):
    definition = ResNetV2ModelDefinition
    def __init__(self, model_url: str = 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4'):
        list_of_urls = [
            # 50 layers
            'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4',

            # 101 layers
            'https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/4',

            # 152 layers
            'https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/4'
        ]
        self.validate_model_url(model_url, list_of_urls)
        self.init(model_url)
        self.vector_length = 2048
