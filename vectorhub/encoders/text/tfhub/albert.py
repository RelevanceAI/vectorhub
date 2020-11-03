from .bert import Bert2Vec
from ....import_utils import *
from ....models_dict import MODEL_REQUIREMENTS
from ....doc_utils import ModelDefinition
from datetime import date
if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-text-tfhub-albert']):
    from official.nlp.bert import tokenization

AlbertModelDefinition = ModelDefinition(
    model_id='text/albert',
    model_name='Albert - A Lite Bert', 
    vector_length=768,
    description="""Increasing model size when pretraining natural language representations often results in improved performance on downstream tasks. However, at some point further model increases become harder due to GPU/TPU memory limitations and longer training times. To address these problems, we present two parameter-reduction techniques to lower memory consumption and increase the training speed of BERT. Comprehensive empirical evidence shows that our proposed methods lead to models that scale much better compared to the original BERT. We also use a self-supervised loss that focuses on modeling inter-sentence coherence, and show it consistently helps downstream tasks with multi-sentence inputs. As a result, our best model establishes new state-of-the-art results on the GLUE, RACE, and \squad benchmarks while having fewer parameters compared to BERT-large.""",
    paper='https://arxiv.org/abs/1909.11942',
    repo='https://tfhub.dev/tensorflow/albert_en_base/1',
    installation="pip install vectorhub[encoders-text-tfhub]",
    release_date=date(2019,9,26),
    example="""
    #pip install vectorhub[encoders-text-tfhub]
    #FOR WINDOWS: pip install vectorhub[encoders-text-tfhub-windows]
    from vectorhub.encoders.text.tfhub import Albert2Vec
    model = Albert2Vec()
    model.encode("I enjoy taking long walks along the beach with my dog.")
    """
)

__doc__ = AlbertModelDefinition.create_docs()

class Albert2Vec(Bert2Vec):
    definition = AlbertModelDefinition
    def __init__(self, model_url: str = 'https://tfhub.dev/tensorflow/albert_en_base/1', max_seq_length: int = 128, normalize: bool = True):
        list_of_urls = [
            'https://tfhub.dev/tensorflow/albert_en_base/1',
            'https://tfhub.dev/tensorflow/albert_en_xxlarge/1',
            'https://tfhub.dev/tensorflow/albert_en_large/1',
            'https://tfhub.dev/tensorflow/albert_en_xlarge/1',
        ]
        self.validate_model_url(model_url, list_of_urls)
        self.max_seq_length = max_seq_length
        self.normalize = normalize
        self.model = self.init(model_url)
        self.tokenizer = self.init_tokenizer()

    def init_tokenizer(self):
        sp_model_file = self.model_layer.resolved_object.sp_model_file.asset_path.numpy()
        return tokenization.FullSentencePieceTokenizer(sp_model_file)
