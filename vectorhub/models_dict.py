"""
    Dictionary For Models
"""

# Map Model to the requirements here.
# This is used to allow the user to list what models they have which can be installed.

MODEL_REQUIREMENTS = {
    "encoders-audio-pytorch-fairseq": "encoders-audio-pytorch",
    "encoders-audio-tfhub-speech_embedding": "encoders-audio-tfhub",
    "encoders-audio-tfhub-trill": 'encoders-audio-tfhub',
    "encoders-audio-tfhub-vggish": "encoders-audio-tfhub",
    "encoders-audio-tfhub-yamnet": "encoders-audio-tfhub", 
    "encoders-audio-vectorai": None,

    "encoders-text-tfhub-albert": "encoders-text-tfhub",
    "encoders-text-tfhub-bert": "encoders-text-tfhub-windows",
    "encoders-text-tfhub-elmo": "encoders-text-tfhub",
    "encoders-text-tfhub-labse": "encoders-text-tfhub-windows",
    "encoders-text-tfhub-use": "encoders-text-tfhub-windows", 
    "encoders-text-tfhub-use-multi": "encoders-text-tfhub", 
    "encoders-text-torch-transformers-legalbert": "encoders-text-torch-transformers",
    "encoders-text-tf-transformers-auto": "encoders-text-tf-transformers",
    "encoders-text-torch-transformers-auto": "encoders-text-torch-transformers",
    "encoders-text-torch-transformers-longformer": "encoders-text-torch-transformers",
    "encoders-text-sentence-transformers": "encoders-text-sentence-transformers",
    "encoders-text-vectorai": None,
    
    
    "encoders-image-tfhub-bit": "encoders-image-tfhub",
    "encoders-image-tfhub-resnet": "encoders-image-tfhub",
    "encoders-image-tfhub-inception": "encoders-image-tfhub",
    "encoders-image-tfhub-inception-resnet": "encoders-image-tfhub",
    "encoders-image-tfhub-mobilenet": "encoders-image-tfhub",
    "encoders-image-fastai-resnet": "encoders-image-fastai",
    "encoders-image-vectorai": None,
    "encoders-image-tf-face-detection": "encoders-image-tf-face-detection",
    
    "text-bi-encoder-tfhub-lareqa-qa": "encoders-text-tfhub",
    "text-bi-encoder-tfhub-use-qa": "encoders-text-tfhub",
    "text-bi-encoder-torch-dpr": "encoders-text-torch-transformers",
    
    "encoders-code-transformers": "encoders-text-transformers",

    "text-image-bi-encoder-clip": "clip"
}
