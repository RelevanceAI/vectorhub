from ...import_utils import *
if is_all_dependency_installed('encoders-image-tfhub'):
    import io
    import imageio
    from urllib.request import urlopen, Request
    from urllib.parse import quote
    import numpy as np

from ...base import Base2Vec
from typing import Union

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
