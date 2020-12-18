"""
    Utilities for importing libraries.
"""
import sys
import warnings
import pkg_resources
import json
from importlib import import_module, invalidate_caches

def get_package_requirements(requirement_type: str):
    """
        Load in extra_requirements.json from the package
    """
    requirements = {
        "numpy": ["core"],
        "requests": ["core"],
        "PyYAML": ["core"],
        "pytest": ["test"],
        "sphinx-rtd-theme>=0.5.0": ["test"],
        "imageio": ["encoders-image", "encoders-image-tfhub"],
        "scikit-image":  ["encoders-image", "encoders-image-tfhub"],
        "soundfile": ["encoders-audio-tfhub"],
        "librosa": ["audio-encoder", "encoders-audio-tfhub"],
        "tensorflow": ["encoders-text-tfhub", "encoders-audio-tfhub", "encoders-image-tfhub", "encoders-text-tf-transformers", "encoders-text-tfhub-windows"],
        "tensorflow_hub": ["encoders-text-tfhub", "encoders-audio-tfhub", "encoders-image-tfhub", "encoders-text-tfhub-windows"],
        "tensorflow_text": ["encoders-text-tfhub"],
        "tf-models-official": ["encoders-text-tfhub", "encoders-text-tfhub-windows"],
        "bert-for-tf2": ["encoders-text-tfhub", "encoders-text-tfhub-windows"],
        "sentence-transformers": ["encoders-text-sentence-transformers"],
        "torch>=1.6.0": ["encoders-audio-pytorch", "encoders-text-torch-transformers", "encoders-text-sentence-transformers", "encoders-image-fastai"],
        "fairseq": ["encoders-audio-pytorch"],
        "transformers": ["encoders-text-torch-transformers", "encoders-text-tf-transformers"],
        "moviepy": ["encoders-video"],
        "opencv-python": ["encoders-video"],
        "fastai==2.1.8": ["encoders-image-fastai"]
    }
    dependencies = []
    for k, v in requirements.items():
        if requirement_type in v:
            dependencies.append(k) 
    return dependencies

def is_dependency_installed(dependency: str):
    """
        Returns True if the dependency is installed else False.
    """
    IS_INSTALLED = True
    try:
        pkg_resources.get_distribution(dependency)
    except pkg_resources.ContextualVersionConflict:
        IS_INSTALLED = True
    except:
        IS_INSTALLED = False
    return IS_INSTALLED

def is_all_dependency_installed(requirement_type: str, raise_warning=True):
    """
        Returns True/False if the dependency is isntalled
        Args:
            requirement_type: The type of requirement. This can be found in the values in extra_requirements.json
            raise_warning: Raise warning if True
    """
    IS_ALL_INSTALLED = True
    requirements = get_package_requirements(requirement_type)
    for r in requirements:
        if not is_dependency_installed(r):
            if raise_warning:
                warnings.warn(f"You are missing {r} dependency for this submodule. Run `pip install vectorhub[{requirement_type}]`")
            IS_ALL_INSTALLED = False
    return IS_ALL_INSTALLED
