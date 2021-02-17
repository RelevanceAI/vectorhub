from datetime import date
from ..base import BaseText2Vec
from ....base import catch_vector_errors
from ....import_utils import is_all_dependency_installed
from ....models_dict import MODEL_REQUIREMENTS
from ....doc_utils import ModelDefinition

if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-text-tfhub-albert']):
    from tensorflow.python.framework.errors_impl import NotFoundError
    import tensorflow as tf
    import tensorflow_hub as hub
    try:
        import tensorflow_text
    except NotFoundError:
        print('The installed Tensorflow Text version is not aligned with tensorflow, make sure that tensorflow-text version is same version as tensorflow')

ExpertsBertModelDefinition = ModelDefinition(markdown_filepath='encoders/text/tfhub/experts_bert')
__doc__ = ExpertsBertModelDefinition.create_docs()

class ExpertsBert2Vec(BaseText2Vec):
    definition = ExpertsBertModelDefinition
    urls = {
        "https://tfhub.dev/google/experts/bert/wiki_books/2": {"vector_length": 768},  
        "https://tfhub.dev/google/experts/bert/wiki_books/mnli/2": {"vector_length": 768},  
        "https://tfhub.dev/google/experts/bert/wiki_books/qnli/2": {"vector_length": 768},  
        "https://tfhub.dev/google/experts/bert/wiki_books/qqp/2": {"vector_length": 768},  
        "https://tfhub.dev/google/experts/bert/wiki_books/sst2/2": {"vector_length": 768},  
        "https://tfhub.dev/google/experts/bert/wiki_books/squad2/2": {"vector_length": 768},  
        "https://tfhub.dev/google/experts/bert/pubmed/2": {"vector_length": 768},  
        "https://tfhub.dev/google/experts/bert/pubmed/squad2/2": {"vector_length": 768},  
    }
    def __init__(self, model_url: str =  "https://tfhub.dev/google/experts/bert/wiki_books/2", max_seq_length: int = 228, normalize: bool = True, 
        preprocessor_url:str ='https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'):
        self.validate_model_url(model_url, list(self.urls.keys()))
        self.max_seq_length = max_seq_length
        self.normalize = normalize
        self.init(model_url)
        self.init_tokenizer(preprocessor_url)

    def init_tokenizer(self, preprocessor_url):
        self.preprocessor = hub.KerasLayer(preprocessor_url)

    def init(self, model_url):
        self.model = hub.KerasLayer(model_url)

    @catch_vector_errors
    def encode(self, text: str, pooling_strategy='pooled_output'):
        return self.model(self.preprocessor([text]))[pooling_strategy].numpy().tolist()[0]

    @catch_vector_errors
    def bulk_encode(self, texts: list, pooling_strategy='pooled_output'):
        return self.model(self.preprocessor(texts))[pooling_strategy].numpy().tolist()
