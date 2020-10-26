from ..base import BaseText2Vec
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import *
from ....models_dict import MODEL_REQUIREMENTS
if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-text-tfhub-use']):
    import tensorflow_hub as hub

USEModelDefinition = ModelDefinition(
    model_id = "text/use",
    model_name="USE - Universal Sentence Encoder", 
    vector_length=512, 
    description="We present models for encoding sentences into embedding vectors that specifically target transfer learning to other NLP tasks. The models are efficient and result in accurate performance on diverse transfer tasks. Two variants of the encoding models allow for trade-offs between accuracy and compute resources. For both variants, we investigate and report the relationship between model complexity, resource consumption, the availability of transfer task training data, and task performance. Comparisons are made with baselines that use word level transfer learning via pretrained word embeddings as well as baselines do not use any transfer learning. We find that transfer learning using sentence embeddings tends to outperform word level transfer. With transfer learning via sentence embeddings, we observe surprisingly good performance with minimal amounts of supervised training data for a transfer task. We obtain encouraging results on Word Embedding Association Tests (WEAT) targeted at detecting model bias. Our pre-trained sentence encoding models are made freely available for download and on TF Hub.",
    paper="https://arxiv.org/abs/1803.11175",
    repo="https://tfhub.dev/google/collections/universal-sentence-encoder/1",
    installation="pip install vectorhub[encoders-text-tfhub]",
    example="""
    #pip install vectorhub[encoders-text-tfhub]
    #FOR WINDOWS: pip install vectorhub[encoders-text-tfhub-windows]
    from vectorhub.encoders.text.tfhub import USE2Vec
    model = USE2Vec()
    model.encode("I enjoy taking long walks along the beach with my dog.")
    """
)

__doc__ = USEModelDefinition.create_docs()

class USE2Vec(BaseText2Vec):
    definition = USEModelDefinition
    # or layer19
    def __init__(self, model_url: str = 'https://tfhub.dev/google/universal-sentence-encoder/4'):
        list_of_urls = [
            "https://tfhub.dev/google/universal-sentence-encoder/4",
            "https://tfhub.dev/google/universal-sentence-encoder-large/5",
        ]
        self.validate_model_url(model_url, list_of_urls)
        self.init(model_url)
        self.vector_length = 512

    def init(self, model_url: str):
        self.model_url = model_url
        self.model = hub.load(self.model_url)
        self.model_name = model_url.replace(
            'https://tfhub.dev/google/', '').replace('/', '_')


    @catch_vector_errors
    def encode(self, text):
        return self.model([text]).numpy().tolist()[0]

    # can consider compress in the future
    @catch_vector_errors
    def bulk_encode(self, texts, threads=10, chunks=100):
        return [i for c in self.chunk(texts, chunks) for i in self.model(c).numpy().tolist()]
