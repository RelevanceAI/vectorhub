from typing import Union
from ...base import Base2Vec
from ...import_utils import *
if is_all_dependency_installed('encoders-image-tfhub'):
    import io
    import imageio
    import numpy as np
    from urllib.request import urlopen, Request
    from urllib.parse import quote
    from skimage import transform


class BaseImage2Vec(Base2Vec):
    def read(self, image: str):
        """
            An method to read images. 
            Args:
                image: An image link/bytes/io Bytesio data format.
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
        try:
            return np.array(imageio.imread(b, pilmode="RGB"))
        except:
            return np.array(imageio.imread(b)[:, :, :3])

    def image_resize(self, image_array, width=0, height=0, rescale=0, resize_mode='symmetry'):
        if width and height:
            image_array = transform.resize(image_array, (width, height), mode=resize_mode, preserve_range=True)
        if rescale:
            image_array = transform.rescale(image_array, rescale, preserve_range=True, anti_aliasing=True)
        return np.array(image_array)