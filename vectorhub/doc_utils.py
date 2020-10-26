"""
    Base 2VEC Definition
"""
from .base import BASE_2VEC_DEFINITON 
from collections import defaultdict

BASE_2VEC_DEFINITION = defaultdict(str, {
    "model_name": None,
    "vector_length": None,
    "description": None,
    "paper": None,
    "repo": None,
    "architecture": None,
    "tasks": None,
    "limitations": None,
})

class ModelDefinition:
    def __init__(self, model_id: str=None, model_name: str=None, vector_length: int=None,
    description: str=None, paper: str=None, repo: str=None, architecture: str='Not stated.',
    tasks: str='Not stated.', limitations: str='Not stated.', installation: str='Not stated.', 
    example: str='Not stated.', **kwargs):
        self.model_id = model_id
        self.model_name = model_name
        self.vector_length = vector_length
        self.description = description
        self.paper = paper
        self.repo = repo
        self.architecture = architecture
        self.tasks = tasks
        self.limitations = limitations
        self.installation = installation
        self.example = example
        for k, v in kwargs.items():
            # assert( k in self.__class__.__allowed )
            setattr(self, k, v)

    
    def create_docs(self):
        """
            Create Documentation
        """
        return f"""
**Model Name**: {self.model_name}

**Vector Length**: {self.vector_length}

**Description**: {self.description}

**Paper**: {self.paper}

**Repository**: {self.repo}

**Architecture**: {self.architecture}

**Tasks**: {self.tasks}

**Limitations**: {self.limitations}

**Installation**: {self.installation}

**Example**: 

.. code-block:: python

    {self.example}
        """

    def create_dict(self):
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "vector_length": self.vector_length,
            "description": self.description,
            "paper": self.paper,
            "repo": self.repo,
            "architecture": self.architecture,
            "tasks": self.tasks,
            "limitations": self.limitations,
            "installation" : self.installation,
            "example" : self.example
        }
