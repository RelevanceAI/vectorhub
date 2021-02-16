"""
    Base Text2Vec Model
"""
import warnings
from abc import ABC, abstractmethod
from typing import Union, List, Dict
from ...base import Base2Vec

class BaseText2Vec(Base2Vec, ABC):
    def read(self, text: str):
        """An abstract method to specify the read method to read the data.
        """
        return text
    
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

    def encode(self, model_input, pooling_strategy=None):
        """
        Note: pooling_strategy only works if the forward method is giving all the outputs, otherwise,
        it only uses the forward method.
        """
        if pooling_strategy is None:
            return self.convert_encode_output_to_list(self.forward(model_input))
        else:
            return self.convert_encode_output_to_list(self.forward(model_input)[pooling_strategy])
    
    @abstractmethod
    def pooling_strategies(self):
        pass

    def bulk_encode(self, model_input, pooling_strategy=None):
        if pooling_strategy is None:
            return self.convert_bulk_encode_output_to_list(self.forward(model_input))
        else:
            return self.convert_bulk_encode_output_to_list(self.forward(model_input)[pooling_strategy])
