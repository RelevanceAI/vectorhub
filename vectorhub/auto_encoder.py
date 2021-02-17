"""
Class for AutoEncoders.
"""
import warnings
from collections import defaultdict
from .models_dict import MODEL_REQUIREMENTS
import json

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    from .encoders.text.tfhub import *
    from .encoders.text.vectorai import *
    from .encoders.text.tf_transformers import *
    from .encoders.text.torch_transformers import *
    from .encoders.text.sentence_transformers import *
    from .encoders.audio.vectorai import *
    from .encoders.audio.tfhub import *
    from .encoders.audio.pytorch import *
    from .encoders.image.tfhub import *
    from .encoders.image.vectorai import *
    from .encoders.image.fastai import *
    from .encoders.face.tf import *
    from .encoders.code.transformers import *
    from .bi_encoders.qa.tfhub import *
    from .bi_encoders.qa.torch_transformers import *
    from .bi_encoders.qa.sentence_transformers import *
    from .bi_encoders.text_image.torch import *

# Include the class and then the requirements key from models_dict.py
ENCODER_MAPPINGS = defaultdict(tuple, {
    # Text models
    Albert2Vec.definition.model_id : (Albert2Vec, "encoders-text-tfhub-albert"),
    ExpertsBert2Vec.definition.model_id : (ExpertsBert2Vec, "encoders-text-tfhub-experts-bert"),
    Bert2Vec.definition.model_id : (Bert2Vec, "encoders-text-tfhub-bert"),
    Elmo2Vec.definition.model_id: (Elmo2Vec, "encoders-text-tfhub-elmo"),
    LaBSE2Vec.definition.model_id : (LaBSE2Vec, "encoders-text-tfhub-labse"),
    LegalBert2Vec.definition.model_id: (LegalBert2Vec, "encoders-text-torch-transformers-legalbert"),
    Longformer2Vec.definition.model_id: (Longformer2Vec, "encoders-text-torch-transformers-longformer"),
    USE2Vec.definition.model_id : (USE2Vec, "encoders-text-tfhub-use"),
    USEMulti2Vec.definition.model_id : (USEMulti2Vec, "encoders-text-tfhub-use"),
    USELite2Vec.definition.model_id : (USELite2Vec, "encoders-text-tfhub-use"),
    

    # "text/tf-transformers" : (TFTransformer2Vec, "encoders-text-tf-transformers"),
    # "text/torch-transformers" : (Transformer2Vec, "encoders-text-torch-transformers"),

    # Audio models
    Wav2Vec.definition.model_id : (Wav2Vec, "encoders-audio-pytorch-fairseq"),
    SpeechEmbedding2Vec.definition.model_id : (SpeechEmbedding2Vec, "encoders-audio-tfhub-speech_embedding"),
    Trill2Vec.definition.model_id : (Trill2Vec, 'encoders-audio-tfhub-trill'),
    TrillDistilled2Vec.definition.model_id : (TrillDistilled2Vec, 'encoders-audio-tfhub-trill'),
    Vggish2Vec.definition.model_id : (Vggish2Vec, 'encoders-audio-tfhub-vggish'),
    Yamnet2Vec.definition.model_id : (Yamnet2Vec, "encoders-audio-tfhub-vggish"),

    # Image models
    BitSmall2Vec.definition.model_id : (BitSmall2Vec, "encoders-image-tfhub-bit"),
    BitMedium2Vec.definition.model_id : (BitMedium2Vec, "encoders-image-tfhub-bit"),
    InceptionV12Vec.definition.model_id : (InceptionV12Vec, "encoders-image-tfhub-inception"),
    InceptionV22Vec.definition.model_id : (InceptionV22Vec, "encoders-image-tfhub-inception"),
    InceptionV32Vec.definition.model_id : (InceptionV32Vec, "encoders-image-tfhub-inception"),
    InceptionResnet2Vec.definition.model_id : (InceptionResnet2Vec, "encoders-image-tfhub-inception-resnet"),
    MobileNetV12Vec.definition.model_id : (MobileNetV12Vec, "encoders-image-tfhub-mobilenet"),
    MobileNetV22Vec.definition.model_id : (MobileNetV22Vec, "encoders-image-tfhub-mobilenet"),
    ResnetV12Vec.definition.model_id : (ResnetV12Vec, "encoders-image-tfhub-resnet"),
    ResnetV22Vec.definition.model_id : (ResnetV22Vec, "encoders-image-tfhub-resnet"),
    FastAIResnet2Vec.definition.model_id: (FastAIResnet2Vec, "encoders-image-fastai-resnet"),

    # Face Model
    Face2Vec.definition.model_id: (Face2Vec, "encoders-image-tf-face-detection"),

    # Code model 
    Code2Vec.definition.model_id: (Code2Vec, "encoders-text-tranfsormers"),
})

class AutoEncoder:
    """
        The AutoEncoder class. To view the AutoEncoder's model, this means 

        Example: 
            >>> from vectorhub.auto_encoder import *
            >>> model = AutoEncoder.from_model('text/bert')
            >>> # To view a list of models use the follow 
            >>> list_all_auto_models()
    """
    def __init__(self):
        pass

    @staticmethod
    def from_model(model_id, *args, **kwargs):
        """
            The AutoEncoder class. To view the AutoEncoder's model, this means 
            Args:
                model_id: The ID of the model - these can be found in list_all_auto_models()
                args: The arguments for the model instantiation
                kwargs: The keyword arguments for the model instantiation
            Example: 
                >>> from vectorhub.auto_encoder import *
                >>> model = AutoEncoder.from_model('text/bert')
                >>> # To view a list of models use the follow 
                >>> list_all_auto_models()
        """
        model_callable, requirements = ENCODER_MAPPINGS[model_id]
        assert is_all_dependency_installed(MODEL_REQUIREMENTS[requirements]), "Missing requirements! Please install."
        model = model_callable(*args, **kwargs)
        return model

BIENCODER_MAPPINGS = {
    USEMultiQA2Vec.definition.model_id : (USEMultiQA2Vec, "text-bi-encoder-tfhub-use-qa"),
    USEQA2Vec.definition.model_id : (USEQA2Vec, "text-bi-encoder-tfhub-use-qa"),
    LAReQA2Vec.definition.model_id: (LAReQA2Vec, "text-bi-encoder-tfhub-lareqa-qa"),
    DPR2Vec.definition.model_id : (DPR2Vec, "text-bi-encoder-torch-dpr"),
    Clip2Vec.definition.model_id: (Clip2Vec, "text-image-clip")
}

class AutoBiEncoder:
    def __init__(self):
        pass
    

    @staticmethod
    def from_model(model_id, *args, **kwargs):
        model_callable, requirements = BIENCODER_MAPPINGS[model_id]
        assert is_all_dependency_installed(MODEL_REQUIREMENTS[requirements]), "Missing requirements! Please install."
        model = model_callable(*args, **kwargs)
        return model


def list_all_auto_models():
    """
        List all available models available for auto models.
    """
    return list(ENCODER_MAPPINGS.keys()) + list(BIENCODER_MAPPINGS.keys())

def get_model_definitions(json_fn='models.json'):
    """
        Get a list of dictionaries with the definitions of the models.
    """
    all_models = []
    global_vars = globals().items()
    for k, v in global_vars:
        if hasattr(v, 'definition'):
            values = v.definition.to_dict()
            values.update({"_id": values['model_id']})
            all_models.append(values)
    if json_fn is not None:
        with open(json_fn, 'w') as f:
            json.dump(all_models, f)
    else:
        return all_models
