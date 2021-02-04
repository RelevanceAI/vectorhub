import warnings
import numpy as np
from datetime import date
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import is_all_dependency_installed
from ....models_dict import MODEL_REQUIREMENTS
from ..base import BaseText2Vec
from .use import USE2Vec

if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-text-tfhub-use-multi']):
    import tensorflow as tf
    from tensorflow.python.framework.errors_impl import NotFoundError
    try:
        import tensorflow_text
    except NotFoundError:
        print('The installed Tensorflow Text version is not aligned with tensorflow, make sure that tensorflow-text version is same version as tensorflow')

USELiteModelDefinition = ModelDefinition(markdown_filepath='encoders/text/tfhub/use_lite.md')

class USELite2Vec(BaseText2Vec):
    definition = USELiteModelDefinition

    urls = {
        "https://tfhub.dev/google/universal-sentence-encoder-lite/2": {'vector_length': 512}
    }
    def __init__(self, model_url: str = 'https://tfhub.dev/google/universal-sentence-encoder-lite/2'):
        list_of_urls = [
            "https://tfhub.dev/google/universal-sentence-encoder-lite/2",
        ]
        self.validate_model_url(model_url, list_of_urls)
        self.vector_length = 512
        warnings.warn("Using USELite2Vec requires disabling tf2 behaviours: tf.disable_v2_behavior(). Meaning it can break the usage of other models if ran. If you are ok with this run model.init() to disable tf2 and run USELite2Vec")

    def init(self):
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        self.input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])
        self.model = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-lite/2")
        self.encodings = module(inputs=dict(
            values=self.input_placeholder.values,
            indices=self.input_placeholder.indices,
            dense_shape=self.input_placeholder.dense_shape
        ))
        with tf.Session() as sess:
            spm_path = sess.run(module(signature="spm_path"))
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spm_path)

    def process_texts(self, texts):
        ids = [self.sp.EncodeAsIds(x) for x in texts]
        return (
            [item for sublist in ids for item in sublist], 
            [[row,col] for row in range(len(ids)) for col in range(len(ids[row]))], 
            (len(ids), max(len(x) for x in ids))
        )

    @catch_vector_errors
    def encode(self, text):
        values, indices, dense_shape = self.process_texts([text])
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            message_embeddings = session.run(self.encodings,
                feed_dict={self.input_placeholder.values: values,
                            self.input_placeholder.indices: indices,
                            self.input_placeholder.dense_shape: dense_shape})
        return np.array(message_embeddings)[0].tolist()

    @catch_vector_errors
    def bulk_encode(self, texts, threads=10, chunks=100):
        values, indices, dense_shape = self.process_texts(texts)
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            message_embeddings = session.run(self.encodings,
                feed_dict={self.input_placeholder.values: values,
                            self.input_placeholder.indices: indices,
                            self.input_placeholder.dense_shape: dense_shape})
        return np.array(message_embeddings).tolist()
