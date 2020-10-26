from ....doc_utils import ModelDefinition
from ....import_utils import *
from ....models_dict import MODEL_REQUIREMENTS
if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-text-tfhub-labse']):
    import tensorflow as tf


LABSEModelDefinition = ModelDefinition(
    model_name="LaBSE - Language-agnostic BERT Sentence Embedding", 
    vector_length=768, 
    description="The language-agnostic BERT sentence embedding encodes text into high dimensional vectors. The model is trained and optimized to produce similar representations exclusively for bilingual sentence pairs that are translations of each other. So it can be used for mining for translations of a sentence in a larger corpus.",
    paper="https://arxiv.org/pdf/2007.01852v1.pdf", 
    repo="https://tfhub.dev/google/LaBSE/1",
    model_id = "text/labse",
    installation="pip install vectorhub[encoders-text-tfhub]",
    example="""
    #pip install vectorhub[encoders-text-tfhub]
    #FOR WINDOWS: pip install vectorhub[encoders-text-tfhub-windows]
    from vectorhub.encoders.text.tfhub import LaBSE2Vec
    model = LaBSE2Vec()
    model.encode("I enjoy taking long walks along the beach with my dog.")
    """
)

__doc__ = LABSEModelDefinition.create_docs()


from .bert import Bert2Vec
class LaBSE2Vec(Bert2Vec):
    definition = LABSEModelDefinition
    def __init__(self, model_url: str = 'https://tfhub.dev/google/LaBSE/1', max_seq_length: int = 128, normalize: bool = True):
        self.max_seq_length = max_seq_length
        self.normalize = normalize
        self.model = self.init(model_url)
        self.tokenizer = self.init_tokenizer()
        self.vector_length = 768
