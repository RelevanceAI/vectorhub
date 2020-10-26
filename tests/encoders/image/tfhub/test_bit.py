import numpy as np
from vectorhub.encoders.image.tfhub import BitMedium2Vec, BitSmall2Vec

def test_bit_initialize():
    """
    Testing for initialize bit model
    """
    client = BitMedium2Vec()
    assert True


def test_bit_encode():
    """
    Testing for bit model single encode
    """
    client = BitMedium2Vec()
    image = client.read('https://getvectorai.com/assets/logo-square.png')

    encoding = client.encode(image)
    assert np.array(encoding).shape == (2048,)


def test_bit_bulk_encode():
    """
    Testing for bit model bulk encode
    """
    client = BitMedium2Vec()
    images_list = [
        'https://getvectorai.com/assets/logo-square.png',
        'https://getvectorai.com/assets/logo-square.png'
    ]

    samples = [client.read(c) for c in images_list]
    encodings = client.bulk_encode(samples)
    assert np.array(encodings).shape == (
        np.array(images_list).shape[0], 2048)


def test_bit_small_initialize():
    """
    Testing for bit small initialize
    """
    client = BitSmall2Vec()
    assert True
