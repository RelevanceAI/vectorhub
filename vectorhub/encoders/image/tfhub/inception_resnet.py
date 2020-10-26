from ..base import BaseImage2Vec
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import *
if is_all_dependency_installed('encoders-image-tfhub'):
    import tensorflow as tf
    import tensorflow_hub as hub
    import traceback

InceptionResnetModelDefinition = ModelDefinition(
    model_id = "image/inception-resnet",
    model_name="Inception Resnet", 
    vector_length=1536, 
    description="""
Very deep convolutional networks have been central to the largest advances in image recognition performance in 
recent years. One example is the Inception architecture that has been shown to achieve very good performance at 
relatively low computational cost. Recently, the introduction of residual connections in conjunction with a more traditional 
architecture has yielded state-of-the-art performance in the 2015 ILSVRC challenge; its performance was similar to the latest 
generation Inception-v3 network. This raises the question of whether there are any benefit in combining the Inception architecture 
with residual connections. Here we give clear empirical evidence that training with residual connections accelerates the training 
of Inception networks significantly. There is also some evidence of residual Inception networks outperforming similarly expensive 
Inception networks without residual connections by a thin margin. We also present several new streamlined architectures for both 
residual and non-residual Inception networks. These variations improve the single-frame recognition performance on the ILSVRC 2012 
classification task significantly. We further demonstrate how proper activation scaling stabilizes the training of very wide residual 
Inception networks. With an ensemble of three residual and one Inception-v4, we achieve 3.08 percent top-5 error on the test set of the 
ImageNet classification (CLS) challenge.""",
    paper="https://arxiv.org/abs/1602.07261",
    repo="https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/4",
    installation="pip install vectorhub[encoders-image-tfhub]",
    example="""
    #pip install vectorhub[encoders-image-tfhub]
    from vectorhub.encoders.image.tfhub import InceptionResnet2Vec
    model = InceptionResnet2Vec()
    sample = model.read('https://getvectorai.com/assets/hub-logo-with-text.png')
    model.encode(sample)
    """
)

__doc__ = InceptionResnetModelDefinition.create_docs()

class InceptionResnet2Vec(BaseImage2Vec):
    definition = InceptionResnetModelDefinition
    def __init__(self, model_url="https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/4"):
        self.model_url = model_url
        self.model_name = self.model_url.replace(
            'https://tfhub.dev/google/', '').replace('/', '_')
        self.model = hub.load(self.model_url)
        self.vector_length = 1536

    @catch_vector_errors
    def encode(self, image):
        """
        Encode an image using InceptionResnet.

        Example:
            >>> from vectorhub.image.encoder.tfhub import inception_resnet
            >>> model = InceptionResnet2Vec(username, api_key)
            >>> model.encode("Hey!")
        """
        return self.model([image]).numpy().tolist()[0]
    
    @catch_vector_errors
    def bulk_encode(self, images, threads=10, chunks=10):
        return [i for c in self.chunk(images, chunks) for i in self.model(c).numpy().tolist()]
