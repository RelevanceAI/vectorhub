"""
    Base Text2Vec Model
"""
import warnings
from ...base import Base2Vec
from abc import ABC, abstractmethod
from typing import Union, List, Dict

class BaseText2Vec(Base2Vec, ABC):
    def read(self, text: str):
        """An abstract method to specify the read method to read the data.
        """
        pass
    
    @property
    def test_word(self):
        return "dummy word"

    @property
    @abstractmethod
    def vector_length(self):
        pass

    @abstractmethod
    def encode(self, words: Union[List[str]]):
        pass
