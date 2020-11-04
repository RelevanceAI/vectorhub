from datetime import date
class ModelDefinition:
    def __init__(self, model_id: str=None, model_name: str=None, vector_length: int=None,
    description: str=None, paper: str=None, repo: str=None, architecture: str='Not stated.',
    tasks: str='Not stated.', release_date: date=None, limitations: str='Not stated.', installation: str='Not stated.', 
    example: str='Not stated.', **kwargs):
        """
            Model definition.
            Args:
                model_id: the identity of the model. Required for AutoEncoder. 
                model_name: The name of the model
                vector_length: The length of the vector 
                description: The description of the encoder
                paper: The paper which dictates the encoder
                repo: The repository fo the model
                architecture: The architecture of the model. 
                task: The downstream task that the model was trained on
                limitations: The limitations of the encoder
                installation: How to isntall the encoder.
                example: The example of the encoder
        """
        self.model_id = model_id
        self.model_name = model_name
        self.vector_length = vector_length
        self.description = description
        self.paper = paper
        self.repo = repo
        self.architecture = architecture
        self.tasks = tasks
        self.release_date = release_date.__str__() if release_date is not None else None
        self.limitations = limitations
        self.installation = installation
        self.example = example
        for k, v in kwargs.items():
            # assert( k in self.__class__.__allowed )
            setattr(self, k, v)

    
    def create_docs(self):
        """
            Return a string with the RST documentation of the model.
        """
        return f"""
**Model Name**: {self.model_name}

**Vector Length**: {self.vector_length}

**Description**: {self.description}

**Paper**: {self.paper}

**Repository**: {self.repo}

**Architecture**: {self.architecture}

**Tasks**: {self.tasks}

**Release Date**: {self.release_date}

**Limitations**: {self.limitations}

**Installation**: ``{self.installation}``

**Example**: 

.. code-block:: python

    {self.example}
        """

    def create_dict(self):
        """
            Create a dictionary with all the attributes of the model. 
        """
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
            "example" : self.example,
            "release_date": self.release_date
        }
