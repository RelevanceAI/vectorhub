"""
    Vector AI's deployed model. The purpose of this model is to allow developers to easily build encodings and see for themselves
    how the embedding works. These models are selected to work out-of-the-box after testing for their success on our end.

    To get access to Vector AI, we need to use 

    Example:

        >>> from vectorhub.text.encoder.vectorai import ViText2Vec
        >>> model = ViText2Vec(username, api_key)
        >>> model.encode("Hey!")
        >>> model.bulk_encode(["hey", "stranger"])

"""
import io
import base64
import numpy as np
import requests
from abc import abstractmethod
from typing import List, Union
from ..base import BaseText2Vec
from ....base import catch_vector_errors

class ViText2Vec(BaseText2Vec):
    def __init__(self, username, api_key, url=None, collection_name="base"):
        """
            Request for a username and API key from gh.vctr.ai!
        """
        self.username = username
        self.api_key = api_key
        if url:
            self.url = url
        else:
            self.url = "https://api.vctr.ai"
        self.collection_name = collection_name
        self._name = "default"

    @catch_vector_errors
    def encode(self, text: Union[str, List[str]]):
        """
            Convert text to vectors.
        """
        if isinstance(text, str):
            return requests.get(
                url="{}/collection/encode_text".format(self.url),
                params={
                    "username": self.username,
                    "api_key": self.api_key,
                    "collection_name": self.collection_name,
                    "text": text,
                },
            ).json()
        elif isinstance(text, list):
            return self.bulk_encode(text)

    @catch_vector_errors
    def bulk_encode(self, texts: List[str]):
        """
            Bulk convert text to vectors
        """
        return requests.get(
            url="{}/collection/bulk_encode_text".format(self.url),
            params={
                "username": self.username,
                "api_key": self.api_key,
                "collection_name": self.collection_name,
                "texts": texts,
            }
        ).json()

    @property
    def __name__(self):
        if self._name is None:
            return "deployed_text"
        return self._name

    @__name__.setter
    def __name__(self, value):
        self._name = value

    @property
    def vector_length(self):
        return 512
