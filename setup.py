#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import os
import codecs
import sys
import json
import re
from setuptools import setup,find_packages
from collections import defaultdict
from typing import List, Dict
from pathlib import Path


def get_extra_requires(path, add_all=True):
    if '.json' in path:
        try:
            requirements_dict = json.load(open(path, 'r'))
            return dependency_to_requirement(requirements_dict)
        except FileNotFoundError:
            print(f"{path} not found")
            return {}

def dependency_to_requirement(requirements_dict: Dict, add_all=True, add_single_package=True):
    """
        Invert the index from dependency to requirement.
    """
    all_requirements = defaultdict(set)
    for library, dependency in requirements_dict.items():
        for d in dependency:
            all_requirements[d].add(library)
    if add_single_package:
        for k in requirements_dict.keys():
            all_requirements[k] = {k}
    if add_all:
        all_requirements['all'] = set(v for v in requirements_dict.keys())
    return all_requirements

all_deps = get_extra_requires('extra_requirements.json')

# Additional files to include - adding model cards
# package_data = [str(x) for x in Path('vectorhub').rglob('*.md')]
package_data = [str(x) for x in list(Path('vectorhub').rglob("*.md"))]

# Also add the extra_requirements.json file
package_data.append('extra_requirements.json')

version = '1.1.5'

if 'IS_VECTORHUB_NIGHTLY' in os.environ.keys():
    from datetime import datetime
    name = 'vectorhub-nightly'
    version = version + '.' + datetime.today().__str__().replace('-', '.').replace(":", '.').replace(' ', '.')
else:
    name = 'vectorhub'

print(package_data)
setup(
    name=name,
    version=version,
    author="OnSearch Pty Ltd",
    author_email="dev@vctr.ai",
    package_data={'vectorhub': package_data, '': ['extra_requirements.json']},
    include_package_data=True,
    # data_files=[('vectorhub', package_data)], # puts the markdown files in a new directory - not what we want
    description="One liner to encode data into vectors with state-of-the-art models using tensorflow, pytorch and other open source libraries. Word2Vec, Image2Vec, BERT, etc",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="vector, embeddings, machinelearning, ai, artificialintelligence, nlp, tensorflow, pytorch, nearestneighbors, search, analytics, clustering, dimensionalityreduction",
    url="https://github.com/vector-ai/vectorhub",
    license="Apache",
    packages=find_packages(exclude=["tests*"]) + ['.'],
    python_requires=">=3",
    install_requires=list(all_deps['core'].union(all_deps['perf'])),
    extras_require=all_deps,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Manufacturing",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Database",
        "Topic :: Multimedia :: Sound/Audio :: Conversion",
        "Topic :: Multimedia :: Video :: Conversion",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
    ],
)
