import numpy as np
from vectorhub.encoders.image.tfhub import InceptionResnet2Vec


def test_bit_initialize():
    """
    Testing for initialize inception_resnet model
    """
    client = InceptionResnet2Vec()
    assert True


def test_inception_resnet_encode():
    """
    Testing for inception_resnet model single encode
    """
    client = InceptionResnet2Vec()
    image = client.read('https://getvectorai.com/assets/logo-square.png')

    encoding = client.encode(image)
    assert np.array(encoding).shape == (1536,)


def test_inception_resnet_bulk_encode():
    """
    Testing for inception_resnet model bulk encode
    """
    client = InceptionResnet2Vec()
    images_list = [
        'https://getvectorai.com/assets/logo-square.png',
        'https://getvectorai.com/assets/logo-square.png'
    ]

    samples = [client.read(c) for c in images_list]
    encodings = client.bulk_encode(samples)
    assert np.array(encodings).shape == (
        np.array(images_list).shape[0], 1536)
