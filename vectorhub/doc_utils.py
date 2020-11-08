from datetime import date
import yaml
import sys
import re

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

    def _get_yaml(self, f):
        """
            Returns YAML file from Python
            Args:
                f: Get YAML file.
        """
        pointer = f.tell()
        if f.readline() != '---\n':
            f.seek(pointer)
            return ''
        readline = iter(f.readline, '')
        readline = iter(readline.__next__, '---\n') #underscores needed for Python3?
        return ''.join(readline)

    def from_markdown(self, markdown_filepath: str):
        """
            Reads definitions from the markdown.
            Args:
                markdown_filepath: The path of the markdown file.
        """
        # Remove sys.argv, not sure what it was doing
        with open(markdown_filepath, encoding='UTF-8') as f:
            config = list(yaml.load_all(self._get_yaml(f), Loader=yaml.SafeLoader))
            text = f.read()
            self.config = config[0]
            for k,v in self.config.items():
                setattr(self, k, v)
            self.description = text
        self._split_markdown_description(text)

    def _split_markdown_description(self, description: str, SPLITTER: str=r"(\#\#+\ +)|(\n)"):
        """
            Breaks markdown into heading and values.
        """
        # Loops through split markdown 
        # If ## is detected inside string, marks the next
        # string as heading
        # and whatever follows as the value 
        IS_HEADING = False
        value = ''
        heading = None
        SKIP_NEW_LINE = False
        markdown_values = {}
        for x in re.split(SPLITTER, self.description):
            if x is None:
                continue
            if SKIP_NEW_LINE:
                if x == '\n':
                    continue
            if IS_HEADING:
                heading = x
                IS_HEADING = False
                # Skip new line after the heading is declared
                SKIP_NEW_LINE = True
                value = ""
            elif '##' in x:
                if heading is not None:
                    setattr(self, heading, value)
                    markdown_values[heading] = value
                IS_HEADING = True
            else:
                SKIP_NEW_LINE = False
                value += x
                
