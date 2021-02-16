import functools
import warnings
import traceback
import numpy as np
import requests
from typing import Any, List
from abc import ABC, abstractmethod
from functools import singledispatchmethod
from .pooler import Pooler
from .reader import Reader
from .typesetter import TypeSetter
from .indexer import ViIndexer
from .errors import ModelError

BASE_2VEC_DEFINITON = {
    "vector_length": None,
    "description": None,
    "paper": None,
    "repo": None,
    "model_name": None,
    "architecture": None,
    "tasks": None,
    "limitations": None,
    "download_required": None,
    "training_required": None,
    "finetunable": None,
}

def catch_vector_errors(func):
    """
        Decorate function and avoid vector errors.
        Example:
            class A:
                @catch_vector_errors
                def encode(self):
                    return [1, 2, 3]
    """
    @functools.wraps(func)
    def catch_vector(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            warnings.warn("Unable to encode. Filling in with dummy vector.")
            traceback.print_exc()
            # get the vector length from the self body
            vector_length = args[0].vector_length
            if isinstance(args[1], str):
                return [1e-7] * vector_length
            elif isinstance(args[1], list):
                # Return the list of vectors
                return [[1e-7] * vector_length] * len(args[1])
            else:
                return [1e-7] * vector_length
    return catch_vector

class Base2Vec(ViIndexer, Pooler, Reader, TypeSetter):
    """
        Base class for vector
    """
    def __init__(self):
        self.__dict__.update(BASE_2VEC_DEFINITON)

    @classmethod
    def validate_model_url(cls, model_url: str, list_of_urls: List[str]):
        """
            Validate the model url belongs in the list of urls. This is to help
            users to avoid mis-spelling the name of the model.

            # TODO:
            Improve model URL validation to not include final number in URl string.

            Args:
                model_url: The URl of the the model in question
                list_of_urls: The list of URLS for the model in question

        """
        if model_url in list_of_urls:
            return True

        if 'tfhub' in model_url:
            # If the url has a number in it then we can take that into account
            for url in list_of_urls:
                if model_url[:-1] in url:
                    return True
        # TODO: Write documentation link to debugging the Model URL.
        warnings.warn("We have not tested this url. Please use URL at your own risk." + \
            "Please use the is_url_working method to test if this is a working url if " + \
            "this is not a local directory.", UserWarning)

    @staticmethod
    def is_url_working(url):
        response = requests.head(url)
        if response.status_code == 200:
            return True
        return False

    @classmethod
    def chunk(self, lst: List, chunk_size: int):
        """
        Chunk an iterable object in Python but not a pandas DataFrame.
        Args:
            lst:
                Python List
            chunk_size:
                The chunk size of an object.
        Example:
            >>> documents = [{...}]
            >>> ViClient.chunk(documents)
        """
        for i in range(0, len(lst), chunk_size):
            yield lst[i: i + chunk_size]

    @singledispatchmethod
    def encode(self, arg):
        raise NotImplementedError("Cannot negate a")

    # @catch_vector_errors
    def encode(self, model_input, pooling_strategy=None):
        """
        Note: pooling_strategy only works if the forward method is giving all the outputs, otherwise,
        it only uses the forward method.
        """
        if pooling_strategy is None:
            return self.convert_encode_output_to_list(self.forward(self.read(model_input)))
        else:
            return self.convert_encode_output_to_list(self.pool(self.forward(self.read(model_input))))
    
    @catch_vector_errors
    def bulk_encode(self, model_inputs):
        if pooling_strategy is None:
            return self.convert_bulk_encode_output_to_list(self.forward(self.bulk_read(model_inputs)))
        else:
            return self.convert_encode_output_to_list(self.pool(self.forward(self.read(model_inputs))))

    @property
    def __name__(self):
        """
            Return the name of the model. If name is not set, returns the
            model_id.
        """
        if hasattr(self, '_name'):
            return self._name.replace('-', '_')
        elif hasattr(self, 'definition'):
            if '/' in self.definition.model_id:
                return self.definition.model_id.split('/')[1].replace('-', '_')
            return self.definition.model_id
        return ''

    @__name__.setter
    def __name__(self, value):
        """
            Set the name.
        """
        setattr(self, '_name', value)
