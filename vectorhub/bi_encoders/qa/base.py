from ...encoders.text.base import BaseText2Vec
from abc import ABC, abstractmethod

class BaseQA2Vec(BaseText2Vec, ABC):
    def encode(self):
        pass

    @abstractmethod
    def encode_question(self):
        pass

    @abstractmethod
    def encode_answer(self):
        pass
