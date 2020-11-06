import numpy as np
from datetime import date
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import *
from ..base import BaseImage2Vec
if is_all_dependency_installed('encoders-image-tfhub'):
    import tensorflow as tf
    import tensorflow_hub as hub

MobileNetModelDefinition = ModelDefinition(
    model_id = "image/mobilenet",
    model_name="MobileNet", 
    vector_length=1024, 
    description="""We present a class of efficient models called MobileNets for mobile and embedded vision applications. MobileNets are based on a streamlined architecture that uses depth-wise separable convolutions to build light weight deep neural networks. We introduce two simple global hyper-parameters that efficiently trade off between latency and accuracy. These hyper-parameters allow the model builder to choose the right sized model for their application based on the constraints of the problem. We present extensive experiments on resource and accuracy tradeoffs and show strong performance compared to other popular models on ImageNet classification. We then demonstrate the effectiveness of MobileNets across a wide range of applications and use cases including object detection, finegrain classification, face attributes and large scale geo-localization.""",
    paper="https://arxiv.org/abs/1704.04861",
    repo="https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/4",
    release_date=date(2017,4,17),
    installation="pip install vectorhub[encoders-image-tfhub]",
    example="""
    #pip install vectorhub[encoders-image-tfhub]
    from vectorhub.encoders.image.tfhub import MobileNetV12Vec
    model = MobileNetV12Vec()
    sample = model.read('https://getvectorai.com/assets/hub-logo-with-text.png')
    model.encode(sample)
    """
)

__doc__ = MobileNetModelDefinition.create_docs()

class MobileNetV12Vec(BaseImage2Vec):
    definition = MobileNetModelDefinition
    def __init__(self, model_url: str = 'https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/4', resize_mode: str='symmetric'):
        list_of_urls = {
            # 100 depth
            'https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/4': {"vector_length":1024, "image_dimensions":224},
            'https://tfhub.dev/google/imagenet/mobilenet_v1_100_192/feature_vector/4': {"vector_length":1024, "image_dimensions":192},
            'https://tfhub.dev/google/imagenet/mobilenet_v1_100_160/feature_vector/4': {"vector_length":1024, "image_dimensions":160},
            'https://tfhub.dev/google/imagenet/mobilenet_v1_100_128/feature_vector/4': {"vector_length":1024, "image_dimensions":128},

            # 75 depth
            'https://tfhub.dev/google/imagenet/mobilenet_v1_075_224/feature_vector/4': {"vector_length":768, "image_dimensions":224},
            'https://tfhub.dev/google/imagenet/mobilenet_v1_075_192/feature_vector/4': {"vector_length":768, "image_dimensions":192},
            'https://tfhub.dev/google/imagenet/mobilenet_v1_075_160/feature_vector/4': {"vector_length":768, "image_dimensions":160},
            'https://tfhub.dev/google/imagenet/mobilenet_v1_075_128/feature_vector/4': {"vector_length":768, "image_dimensions":128},

            # 50 depth
            'https://tfhub.dev/google/imagenet/mobilenet_v1_050_224/feature_vector/4': {"vector_length":512, "image_dimensions":224},
            'https://tfhub.dev/google/imagenet/mobilenet_v1_050_192/feature_vector/4': {"vector_length":512, "image_dimensions":192},
            'https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/feature_vector/4': {"vector_length":512, "image_dimensions":160},
            'https://tfhub.dev/google/imagenet/mobilenet_v1_050_128/feature_vector/4': {"vector_length":512, "image_dimensions":128},

            # 25 depth
            'https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/feature_vector/4': {"vector_length":256, "image_dimensions":224},
            'https://tfhub.dev/google/imagenet/mobilenet_v1_025_192/feature_vector/4': {"vector_length":256, "image_dimensions":192},
            'https://tfhub.dev/google/imagenet/mobilenet_v1_025_160/feature_vector/4': {"vector_length":256, "image_dimensions":160},
            'https://tfhub.dev/google/imagenet/mobilenet_v1_025_128/feature_vector/4': {"vector_length":256, "image_dimensions":128},
        }
        self.validate_model_url(model_url, list_of_urls)
        self.vector_length = list_of_urls[model_url]["vector_length"]
        self.image_dimensions = list_of_urls[model_url]["image_dimensions"]
        self.init(model_url)
        self.resize_mode = resize_mode

    def init(self, model_url: str):
        self.model_url = model_url
        self.model_name = self.model_url.replace(
            'https://tfhub.dev/google/', '').replace('/', '_')
        self.model = tf.keras.Sequential([
            hub.KerasLayer(self.model_url, trainable=False)
        ])
        self.model.build([None, self.image_dimensions, self.image_dimensions, 3])
    
    @catch_vector_errors
    def encode(self, image):
        return self.model(
            self.image_resize(image, self.image_dimensions, self.image_dimensions, resize_mode=self.resize_mode)[np.newaxis, ...]
        ).numpy().tolist()[0]
    
    @catch_vector_errors
    def bulk_encode(self, images, threads=10, chunks=10):
        return [i for c in self.chunk(images, chunks) for i in self.model(c).numpy().tolist()]

MobileNetV2ModelDefinition = ModelDefinition(
    model_id = "image/mobilenet-v2",
    model_name="MobileNet", 
    vector_length=1792, 
    description="""We present a class of efficient models called MobileNets for mobile and embedded vision applications. MobileNets are based on a streamlined architecture that uses depth-wise separable convolutions to build light weight deep neural networks. We introduce two simple global hyper-parameters that efficiently trade off between latency and accuracy. These hyper-parameters allow the model builder to choose the right sized model for their application based on the constraints of the problem. We present extensive experiments on resource and accuracy tradeoffs and show strong performance compared to other popular models on ImageNet classification. We then demonstrate the effectiveness of MobileNets across a wide range of applications and use cases including object detection, finegrain classification, face attributes and large scale geo-localization.""",
    paper="https://arxiv.org/abs/1704.04861",
    repo="https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/4",
    installation="pip install vectorhub[encoders-image-tfhub]",
    example="""
    #pip install vectorhub[encoders-image-tfhub]
    from vectorhub.encoders.image.tfhub import MobileNetV22Vec
    model = MobileNetV22Vec()
    sample = model.read('https://getvectorai.com/assets/hub-logo-with-text.png')
    model.encode(sample)
    """
)

class MobileNetV22Vec(MobileNetV12Vec):
    definition = MobileNetV2ModelDefinition
    def __init__(self, model_url: str = 'https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4', resize_mode:str="symmetric"):
        list_of_urls = {
            # 140 depth
            'https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4': {"vector_length":1792, "image_dimensions":224},

            # 130 depth
            'https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/feature_vector/4': {"vector_length":1664, "image_dimensions":224},

            # 100 depth
            'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4': {"vector_length":1280, "image_dimensions":224},
            'https://tfhub.dev/google/imagenet/mobilenet_v2_100_192/feature_vector/4': {"vector_length":1280, "image_dimensions":192},
            'https://tfhub.dev/google/imagenet/mobilenet_v2_100_160/feature_vector/4': {"vector_length":1280, "image_dimensions":160},
            'https://tfhub.dev/google/imagenet/mobilenet_v2_100_128/feature_vector/4': {"vector_length":1280, "image_dimensions":128},
            'https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/4': {"vector_length":1280, "image_dimensions":96},

            # 75 depth
            'https://tfhub.dev/google/imagenet/mobilenet_v2_075_224/feature_vector/4': {"vector_length":1280, "image_dimensions":224},
            'https://tfhub.dev/google/imagenet/mobilenet_v2_075_192/feature_vector/4': {"vector_length":1280, "image_dimensions":192},
            'https://tfhub.dev/google/imagenet/mobilenet_v2_075_160/feature_vector/4': {"vector_length":1280, "image_dimensions":160},
            'https://tfhub.dev/google/imagenet/mobilenet_v2_075_128/feature_vector/4': {"vector_length":1280, "image_dimensions":128},
            'https://tfhub.dev/google/imagenet/mobilenet_v2_075_96/feature_vector/4': {"vector_length":1280, "image_dimensions":96},

            # 50 depth
            'https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/feature_vector/4': {"vector_length":1280, "image_dimensions":224},
            'https://tfhub.dev/google/imagenet/mobilenet_v2_050_192/feature_vector/4': {"vector_length":1280, "image_dimensions":192},
            'https://tfhub.dev/google/imagenet/mobilenet_v2_050_160/feature_vector/4': {"vector_length":1280, "image_dimensions":160},
            'https://tfhub.dev/google/imagenet/mobilenet_v2_050_128/feature_vector/4': {"vector_length":1280, "image_dimensions":128},
            'https://tfhub.dev/google/imagenet/mobilenet_v2_050_96/feature_vector/4': {"vector_length":1280, "image_dimensions":96},

            # 35 depth
            'https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/feature_vector/4': {"vector_length":1280, "image_dimensions":224},
            'https://tfhub.dev/google/imagenet/mobilenet_v2_035_192/feature_vector/4': {"vector_length":1280, "image_dimensions":192},
            'https://tfhub.dev/google/imagenet/mobilenet_v2_035_160/feature_vector/4': {"vector_length":1280, "image_dimensions":160},
            'https://tfhub.dev/google/imagenet/mobilenet_v2_035_128/feature_vector/4': {"vector_length":1280, "image_dimensions":128},
            'https://tfhub.dev/google/imagenet/mobilenet_v2_035_96/feature_vector/4': {"vector_length":1280, "image_dimensions":96},
        }
        self.validate_model_url(model_url, list_of_urls)
        self.vector_length = list_of_urls[model_url]["vector_length"]
        self.image_dimensions = list_of_urls[model_url]["image_dimensions"]
        self.init(model_url)
        self.resize_mode = resize_mode
