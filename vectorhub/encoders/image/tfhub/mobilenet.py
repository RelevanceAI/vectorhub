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

    def init(self, model_url: str):
        self.model_url = model_url
        self.model_name = self.model_url.replace(
            'https://tfhub.dev/google/', '').replace('/', '_')
        self.model = tf.keras.Sequential([
            hub.KerasLayer(self.model_url, trainable=False)
        ])
        self.model.build([None, self.image_dimensions, self.image_dimensions, 3])
    
    def _read(self, image: str):
        """
            An method to read images. 
            Args:
                image: An image link/bytes/io Bytesio data format.
                as_gray: read in the image as black and white
        """
        if type(image) == str:
            if 'http' in image:
                b = io.BytesIO(urlopen(Request(
                    quote(image, safe=':/?*=\''), headers={'User-Agent': "Mozilla/5.0"})).read())
            else:
                b = image
        elif type(image) == bytes:
            b = io.BytesIO(image)
        elif type(image) == io.BytesIO:
            b = image
        else:
            raise ValueError("Cannot process data type. Ensure it is is string/bytes or BytesIO.")
        try:
            return np.array(imageio.imread(b, pilmode="RGB"))
        # TODO: Flesh out exceptions
        except:
            return np.array(imageio.imread(b)[:, :, :3])
    
    def read(self, image, as_mobilenet_input=True):
        """
            Read in the images.
            Args:
                image: The link to the image
                as_mobilenet_input: Reading in the image as MobileNet input
        """
        if as_mobilenet_input:
            return self.image_resize(self._read(image), self.image_dimensions, 
            self.image_dimensions, resize_mode=self.resize_mode)[np.newaxis, ...]
        return self._read(image)
    
    @catch_vector_errors
    def encode(self, image):
        if isinstance(image, str):
            image = self.read(image)
        return self.model(image).numpy().tolist()[0]

    @catch_vector_errors
    def bulk_encode(self, images):
        """
            Bulk encode. Chunk size should be specified outside of the images.
        """
        # TODO: Change from list comprehension to properly read
        return [self.encode(x) for x in images]
