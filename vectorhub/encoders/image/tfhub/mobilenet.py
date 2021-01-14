import numpy as np
from datetime import date
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import *
from ..base import BaseImage2Vec
if is_all_dependency_installed('encoders-image-tfhub'):
    import tensorflow as tf
    import tensorflow_hub as hub
    import io
    import imageio
    import numpy as np
    import matplotlib.pyplot as plt
    from urllib.request import urlopen, Request
    from urllib.parse import quote
    from skimage import transform

MobileNetModelDefinition = ModelDefinition(markdown_filepath='encoders/image/tfhub/mobilenet')

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

    @property
    def urls(self):
        return {
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
        if isinstance(image, str):
            image = self.read(image)
        resized_image = self.image_resize(image, self.image_dimensions, self.image_dimensions,
            resize_mode=self.resize_mode)[np.newaxis, ...]
        return self.model(resized_image).numpy().tolist()[0]

    @catch_vector_errors
    def bulk_encode(self, images):
        """
            Bulk encode. Chunk size should be specified outside of the images.
        """
        # TODO: Change from list comprehension to properly read in bulk
        return [self.encode(x) for x in images]
