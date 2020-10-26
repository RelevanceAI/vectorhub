"""
    Utilities for importing libraries.
"""
import sys
import warnings
import pkg_resources
import json
from pkg_resources import resource_filename
from importlib import import_module, invalidate_caches

def get_requirements(extra_requirements_file: str=resource_filename('vectorhub', '../extra_requirements.json')):
    if '.json' in extra_requirements_file:
        d = json.load(open(extra_requirements_file, 'r'))
        return d

def get_package_requirements(requirement_type):
    requirements = get_requirements()
    dependencies = []
    for k, v in requirements.items():
        if requirement_type in v:
            dependencies.append(k) 
    return dependencies

def is_dependency_installed(dependency):
    IS_INSTALLED = True
    try:
        pkg_resources.get_distribution(dependency)
    except:
        IS_INSTALLED = False
    return IS_INSTALLED

def is_all_dependency_installed(requirement_type, raise_warning=True):
    IS_ALL_INSTALLED = True
    requirements = get_package_requirements(requirement_type)
    for r in requirements:
        if not is_dependency_installed(r):
            if raise_warning:
                warnings.warn(f"You are missing dependencies for this submodule. Run `pip install vectorhub[{requirement_type}]`")
            IS_ALL_INSTALLED = False
    return IS_ALL_INSTALLED
