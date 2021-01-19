"""
    Various utilities for VectorHub.
"""
import json
import os
from pathlib import Path
from pkg_resources import resource_filename
from collections import defaultdict
from .models_dict import *
from .import_utils import *

def list_installed_models(extra_requirements_file: str=resource_filename('vectorhub', '../extra_requirements.json')):
    """
        List models that are installed.
        We use resource_filename to resolve relative directory issues.
    """
    requirements = json.load(open(extra_requirements_file))
    print("The following packages are available to be used: ")
    all_available_models = []
    for package, dependency in MODEL_REQUIREMENTS.items():
        if is_all_dependency_installed(dependency, raise_warning=False):
            print(package)
            all_available_models.append(package)
    return all_available_models

def list_models(return_names_only=False):
    """
        List available models.
        Args:
            return_names_only: Return the model names
    """
    if return_names_only:
        return [x.stem for x in list(Path('.').glob('**/[!_]*.py'))]
    
    all_models = [str(x).replace('.py', '') for x in list(Path(resource_filename('vectorhub', 'encoders/text')).rglob(f'[!_]*.py'))] + \
        [str(x).replace('.py', '') for x in list(Path(resource_filename('vectorhub', 'encoders/image')).rglob(f'[!_]*.py'))] + \
        [str(x).replace('.py', '') for x in list(Path(resource_filename('vectorhub', 'encoders/audio')).rglob(f'[!_]*.py'))] + \
        [str(x).replace('.py', '') for x in list(Path(resource_filename('vectorhub', 'bi_encoders/qa')).rglob(f'[!_]*.py'))]
    return [x.split('vectorhub/')[-1] for x in all_models if '/base' not in x]

