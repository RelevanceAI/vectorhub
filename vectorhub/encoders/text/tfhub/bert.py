from ..base import BaseText2Vec
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import is_all_dependency_installed
from ....models_dict import MODEL_REQUIREMENTS
from datetime import date
if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-text-tfhub-bert']):
    import tensorflow as tf
    if hasattr(tf, 'executing_eagerly'):
        if not tf.executing_eagerly():
            tf.compat.v1.enable_eager_execution()
    import tensorflow_hub as hub
    import bert
    import numpy as np

BertModelDefinition = ModelDefinition(markdown_filepath='encoders/text/tfhub/bert')

__doc__ = BertModelDefinition.create_docs()

class Bert2Vec(BaseText2Vec):
    definition = BertModelDefinition
    def __init__(self, model_url: str = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/3', 
    max_seq_length: int = 64, normalize: bool = True):
        list_of_urls = [
            'https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/2',
            'https://tfhub.dev/tensorflow/bert_en_wwm_cased_L-24_H-1024_A-16/2',
            'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2',
            'https://tfhub.dev/tensorflow/bert_en_wwm_uncased_L-24_H-1024_A-16/2',
            'https://tfhub.dev/tensorflow/bert_en_cased_L-24_H-1024_A-16/2',
            'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/2',
            'https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/2',
            'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/2',

            'https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/3',
            'https://tfhub.dev/tensorflow/bert_en_wwm_cased_L-24_H-1024_A-16/3',
            'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
            'https://tfhub.dev/tensorflow/bert_en_wwm_uncased_L-24_H-1024_A-16/3',
            'https://tfhub.dev/tensorflow/bert_en_cased_L-24_H-1024_A-16/3',
            'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
            'https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/3',
            'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
        ]
        self.validate_model_url(model_url, list_of_urls)
        self.max_seq_length = max_seq_length
        self.normalize = normalize
        self.model_input_type = "dict"
        self.init(model_url)
        self.tokenizer = self.init_tokenizer()

    @property
    def urls(self):
        {
            'https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/2': {'vector_length': 1024},
            'https://tfhub.dev/tensorflow/bert_en_wwm_cased_L-24_H-1024_A-16/2': {'vector_length': 1024},
            'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2': {'vector_length': 768},
            'https://tfhub.dev/tensorflow/bert_en_wwm_uncased_L-24_H-1024_A-16/2': {'vector_length': 1024},
            'https://tfhub.dev/tensorflow/bert_en_cased_L-24_H-1024_A-16/2': {'vector_length': 1024},
            'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/2': {'vector_length': 768},
            'https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/2': {'vector_length': 768},
            'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/2': {'vector_length': 768},

            'https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/3': {'vector_length': 1024},
            'https://tfhub.dev/tensorflow/bert_en_wwm_cased_L-24_H-1024_A-16/3': {'vector_length': 1024},
            'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3': {'vector_length': 768},
            'https://tfhub.dev/tensorflow/bert_en_wwm_uncased_L-24_H-1024_A-16/3': {'vector_length': 1024},
            'https://tfhub.dev/tensorflow/bert_en_cased_L-24_H-1024_A-16/3': {'vector_length': 1024},
            'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3': {'vector_length': 768},
            'https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/3': {'vector_length': 768},
            'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3': {'vector_length': 768},
        }

    def init(self, model_url: str):
        self.model = hub.KerasLayer(model_url)
        input_word_ids = tf.keras.layers.Input(shape=(self.max_seq_length,), dtype=tf.int32)
        input_mask = tf.keras.layers.Input(shape=(self.max_seq_length,), dtype=tf.int32)
        input_type_ids = tf.keras.layers.Input(shape=(self.max_seq_length,), dtype=tf.int32)
        try:
            self.model(dict(input_word_ids=input_word_ids, input_mask=input_mask, input_type_ids=input_type_ids))
        except ValueError:
            self.model_input_type = "list"
            self.model([input_word_ids, input_mask, input_type_ids])

    def init_tokenizer(self):
        self.vocab_file = self.model.resolved_object.vocab_file.asset_path.numpy()
        self.do_lower_case = self.model.resolved_object.do_lower_case.numpy()
        return bert.bert_tokenization.FullTokenizer(self.vocab_file, self.do_lower_case)

    def process(self, input_strings: str):
        input_ids_all, input_mask_all, input_type_ids_all = [], [], []
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
            input_type_ids_all.append([0] * self.max_seq_length)

        return np.array(input_ids_all), np.array(input_mask_all),  np.array(input_type_ids_all)

    @catch_vector_errors
    def encode(self, text: str, pooling_strategy='pooled_output'):
        input_ids, input_mask, input_type_ids = self.process(text)
        if self.model_input_type == "list":
            return self.model([
                tf.convert_to_tensor(input_ids, tf.int32, name="input_word_ids"), 
                tf.convert_to_tensor(input_mask, tf.int32, name="input_mask"), 
                tf.convert_to_tensor(input_type_ids, tf.int32, name="input_type_ids")
            ])[0].numpy().tolist()[0]
        else:
            return self.model({
                "input_word_ids": tf.convert_to_tensor(input_ids, tf.int32, name="input_word_ids"), 
                "input_mask": tf.convert_to_tensor(input_mask, tf.int32, name="input_mask"), 
                "input_type_ids": tf.convert_to_tensor(input_type_ids, tf.int32, name="input_type_ids")
            })[pooling_strategy].numpy().tolist()[0]


    @catch_vector_errors
    def bulk_encode(self, texts: list, pooling_strategy='pooled_output'):
        input_ids, input_mask, input_type_ids = self.process(texts)
        if self.model_input_type == "list":
            return self.model([
                tf.convert_to_tensor(input_ids, tf.int32, name="input_word_ids"), 
                tf.convert_to_tensor(input_mask, tf.int32, name="input_mask"), 
                tf.convert_to_tensor(input_type_ids, tf.int32, name="input_type_ids")
            ])[0].numpy().tolist()
        else:
            return self.model({
                "input_word_ids": tf.convert_to_tensor(input_ids, tf.int32, name="input_word_ids"), 
                "input_mask": tf.convert_to_tensor(input_mask, tf.int32, name="input_mask"), 
                "input_type_ids": tf.convert_to_tensor(input_type_ids, tf.int32, name="input_type_ids")
            })[pooling_strategy].numpy().tolist()
