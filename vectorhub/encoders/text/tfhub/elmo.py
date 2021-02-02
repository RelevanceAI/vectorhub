import warnings
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import *
from ....models_dict import MODEL_REQUIREMENTS
from ..base import BaseText2Vec
from datetime import date
if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-text-tfhub-use']):
    import tensorflow_hub as hub
    import tensorflow.compat.v1 as tf
    import numpy as np

ElmoModelDefinition = ModelDefinition(markdown_filepath='encoders/text/tfhub/elmo.md')

__doc__ = ElmoModelDefinition.create_docs()

class Elmo2Vec(BaseText2Vec):
    definition = ElmoModelDefinition
    urls ={
        "https://tfhub.dev/google/elmo/3": {'vector_length': 1024}
    }
    def __init__(self, model_url: str="https://tfhub.dev/google/elmo/3", trainable_model=True):
        warnings.warn("We are disabling TF2 eager execution to run this. This may conflict with other models. If you need + \
            other models., try to use a fresh environment or a new virtual machine.")
        tf.disable_eager_execution()
        self.model = hub.Module(model_url, trainable=trainable_model)
        self.vector_length = 1024
    
    @catch_vector_errors
    def encode(self, text, output_layer: str="elmo"):
        """
        The output layer can be one of the following:
        lstm_outputs1: the first LSTM hidden state with shape [batch_size, max_length, 1024].
        lstm_outputs2: the second LSTM hidden state with shape [batch_size, max_length, 1024].
        elmo: the weighted sum of the 3 layers, where the weights are trainable. This tensor has shape [batch_size, max_length, 1024]
        default: a fixed mean-pooling of all contextualized word representations with shape [batch_size, 1024].
        Note: The output layer word_emb is character-based and is not supported by VectorHub.
        """
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        if output_layer != 'default':
            vector = self.model(
                [text],
                signature="default",
                as_dict=True)[output_layer].eval(session=sess)[0][0].tolist()
        else:
            vector = self.model(
                [text],
                signature="default",
                as_dict=True)[output_layer].eval(session=sess)[0].tolist()
        sess.close()
        return vector

    @catch_vector_errors
    def bulk_encode(self, texts, output_layer: str="elmo"):
        """
        The output layer can be one of the following:
            lstm_outputs1: the first LSTM hidden state with shape [batch_size, max_length, 1024].
            lstm_outputs2: the second LSTM hidden state with shape [batch_size, max_length, 1024].
            elmo: the weighted sum of the 3 layers, where the weights are trainable. This tensor has shape [batch_size, max_length, 1024]
            default: a fixed mean-pooling of all contextualized word representations with shape [batch_size, 1024].
        Note: The output layer word_emb is character-based and is not supported by VectorHub.
        """
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        vectors = self.model(
            texts,
            signature="default",
            as_dict=True)[output_layer].eval(session=sess).tolist()
        sess.close()
        if output_layer == 'default':
            return vectors
        else:
            return [x[0] for x in vectors]
