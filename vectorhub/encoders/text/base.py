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
    def vector_length(self):
        """
            Set the vector length of the model.
        """
        if hasattr(self, "_vector_length"):
            return getattr(self, "_vector_length") 
        else:
            print(f"The vector length is not explicitly stated so we are inferring " + \
                "from our test word - {self.test_word}.")
            setattr(self, "_vector_length", len(self.encode(self.test_word)))
            return self._vector_length
    
    @vector_length.setter
    def vector_length(self, value):
        self._vector_length = value
    
    @abstractmethod
    def encode(self, words: Union[List[str]]):
        pass