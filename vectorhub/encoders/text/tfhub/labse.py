from datetime import date
from ..base import BaseText2Vec
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import is_all_dependency_installed
from ....models_dict import MODEL_REQUIREMENTS

if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-text-tfhub-labse']):
    import tensorflow as tf
    if hasattr(tf, 'executing_eagerly'):
        if not tf.executing_eagerly():
            tf.compat.v1.enable_eager_execution()
    import tensorflow_hub as hub
    import bert
    import numpy as np

LABSEModelDefinition = ModelDefinition(markdown_filepath='encoders/text/tfhub/labse.md')
__doc__ = LABSEModelDefinition.create_docs()


class LaBSE2Vec(BaseText2Vec):
    definition = LABSEModelDefinition
    def __init__(self, model_url: str = 'https://tfhub.dev/google/LaBSE/1', max_seq_length: int = 128, normalize: bool = True):
        self.max_seq_length = max_seq_length
        self.normalize = normalize
        self.model = self.init(model_url)
        self.tokenizer = self.init_tokenizer()
        self.vector_length = 768

    @property
    def urls(self):
        return {
            'https://tfhub.dev/google/LaBSE/1': {'vector_length': 768}
        }

    def init(self, model_url: str):
        self.model_layer = hub.KerasLayer(model_url)
        input_word_ids = tf.keras.layers.Input(shape=(self.max_seq_length,), dtype=tf.int32,
                                            name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(self.max_seq_length,), dtype=tf.int32,
                                        name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(self.max_seq_length,), dtype=tf.int32,
                                            name="segment_ids")
        pooled_output,  _ = self.model_layer(
            [input_word_ids, input_mask, segment_ids])
        if(self.normalize):
            pooled_output = tf.keras.layers.Lambda(
                lambda x: tf.nn.l2_normalize(x, axis=1))(pooled_output)
        return tf.keras.Model(
            inputs=[input_word_ids, input_mask, segment_ids],
            outputs=pooled_output)

    def init_tokenizer(self):
        self.vocab_file = self.model_layer.resolved_object.vocab_file.asset_path.numpy()
        self.do_lower_case = self.model_layer.resolved_object.do_lower_case.numpy()
        return bert.bert_tokenization.FullTokenizer(
            self.vocab_file, self.do_lower_case)

    def process(self, input_strings: str):
        input_ids_all, input_mask_all, segment_ids_all = [], [], []
        if isinstance(input_strings, str):
            input_strings = [input_strings]
        for input_string in input_strings:
            # Tokenize input.
            input_tokens = ["[CLS]"] + \
                self.tokenizer.tokenize(input_string) + ["[SEP]"]
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            sequence_length = min(len(input_ids), self.max_seq_length)

            # Padding or truncation.
            if len(input_ids) >= self.max_seq_length:
                input_ids = input_ids[:self.max_seq_length]
            else:
                input_ids = input_ids + [0] * \
                    (self.max_seq_length - len(input_ids))

            input_mask = [1] * sequence_length + [0] * \
                (self.max_seq_length - sequence_length)

            input_ids_all.append(input_ids)
            input_mask_all.append(input_mask)
            segment_ids_all.append([0] * self.max_seq_length)

        return np.array(input_ids_all), np.array(input_mask_all), np.array(segment_ids_all)

    @catch_vector_errors
    def encode(self, text: str):
        input_ids, input_mask, segment_ids = self.process(text)
        return self.model([input_ids, input_mask, segment_ids]).numpy().tolist()[0]

    @catch_vector_errors
    def bulk_encode(self, texts: list):
        input_ids, input_mask, segment_ids = self.process(texts)
        return self.model([input_ids, input_mask, segment_ids]).numpy().tolist()