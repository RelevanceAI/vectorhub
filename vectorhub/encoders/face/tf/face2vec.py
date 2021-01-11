"""
    Face2Vec Embedding.
"""
import os
from typing import Union
from datetime import date
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import *
from ...image.base import BaseImage2Vec

if is_all_dependency_installed("encoders-image-tf-face-detection"):
    import tensorflow as tf
    if hasattr(tf, 'executing_eagerly'):
        if not tf.executing_eagerly():
            tf.compat.v1.enable_eager_execution()
    import appdirs
    import cv2
    import numpy as np
    import requests
    from mtcnn.mtcnn import MTCNN
    from numpy import asarray
    from PIL import Image
    from keras.models import load_model

FaceNetModelDefinition = ModelDefinition(markdown_filepath="encoders/face/tf/face2vec")
__doc__ = FaceNetModelDefinition.create_docs()

class Face2Vec(BaseImage2Vec):
    definition = FaceNetModelDefinition
    def __init__(self, model_url: str = 'https://drive.google.com/u/0/uc?id=1PZ_6Zsy1Vb0s0JmjEmVd8FS99zoMCiN1&export=download', redownload=True):
        if not os.path.exists(self.model_path) or redownload:
            self._download_model(model_url)
        self.vector_length = self.urls[model_url]["vector_length"]
        self.model = load_model(self.model_path)

    def _download_model(self, model_url):
        response = requests.get(model_url)
        if response.status_code != 200:
            raise Exception(response.content)
        with open(self.model_path, 'wb') as f:
            f.write(response.content)

    @property
    def model_path(self):
        return os.path.join(self.cache_dir, 'facenet.h5')

    @property
    def cache_dir(self):
        return appdirs.user_cache_dir()

    @property
    def urls(self):
        """
        A simple dictionary with urls and their vector lengths
        """
        return {'https://drive.google.com/u/0/uc?id=1PZ_6Zsy1Vb0s0JmjEmVd8FS99zoMCiN1&export=download': {'vector_length': 128}}

    def extract_face(self, image_input, reshape_size=None):
        if isinstance(image_input, str):
            pixels = self.read(image_input)
        elif isinstance(image_input, np.ndarray):
            pixels = image_input

        # create the detector, using default weights
        detector = MTCNN()
        # detect faces in the image
        results = detector.detect_faces(pixels)

        # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        # extract the face
        face = pixels[y1:y2, x1:x2]

        # resize pixels to the model size
        image = Image.fromarray(face)
        if reshape_size is not None:
            image = image.resize(reshape_size)
        face_array = asarray(image)
        return face_array

    def show_face_landmarks(self, image_filename: str):
        """
        Show face landmarks
        """
        detector = MTCNN()

        # image = cv2.cvtColor(cv2.imread("rose.jpeg"), cv2.COLOR_BGR2RGB)
        image = self.extract_face(image_filename)
        result = detector.detect_faces(image)
        bounding_box = result[0]['box']
        keypoints = result[0]['keypoints']

        cv2.rectangle(image,
        (bounding_box[0], bounding_box[1]),
        (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
        (0,155,255), 2)
        cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
        cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
        cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
        cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
        cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)
        return plt.imshow(image)

    def standardise_image(self, face_pixels):
        """
        Standardise the image for face pixels.
        """
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        return tf.expand_dims(face_pixels, axis=0)

    @catch_vector_errors
    def encode(self, image):
        if isinstance(image, (np.ndarray, str)):
            image = self.standardise_image(self.extract_face(image, reshape_size=(160, 160)))
        return self.model.predict([image]).tolist()[0]

    @catch_vector_errors
    def bulk_encode(self, images):
        """
            Bulk encode. Chunk size should be specified outside of the images.
        """
        # TODO: Change from list comprehension to properly read
        return [self.encode(x) for x in images]
