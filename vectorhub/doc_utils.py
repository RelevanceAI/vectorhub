import yaml
import sys
import re
import os
from datetime import date
from pkg_resources import resource_exists, resource_filename

class ModelDefinition:
    def __init__(self, model_id: str='', model_name: str='', vector_length: int='',
    description: str='', paper: str='', repo: str='', architecture: str='Not stated.',
    tasks: str='Not stated.', release_date: date='', limitations: str='Not stated.', installation: str='Not stated.',
    example: str='Not stated.', markdown_filepath: str='', **kwargs):
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
        self.markdown_filepath = markdown_filepath
        for k, v in kwargs.items():
            # assert( k in self.__class__.__allowed )
            setattr(self, k, v)
        if markdown_filepath != '':
            self.from_markdown(markdown_filepath)


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

    def to_dict(self, return_base_dictionary=False):
        """
            Create a dictionary with all the attributes of the model.
        """
        if return_base_dictionary:
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
        else:
            model_dict = {}
            for attr in dir(self):
                if '__' in attr:
                    continue
                if isinstance(getattr(self, attr), (float, str, int)):
                    model_dict[attr] = getattr(self, attr)
            # Enforce string typecast on vector length
            model_dict['vector_length'] = str(model_dict['vector_length'])
            return model_dict

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

    def from_markdown(self, markdown_filepath: str, encoding='UTF-8', splitter=r"(\#\#+\ +)|(\n)",
    verbose=False):
        """
            Reads definitions from the markdown.
            Args:
                markdown_filepath: The path of the markdown file.
                encoding: The encoding used to open the Markdown file
        """
        if '.md' not in markdown_filepath:
            markdown_filepath += '.md'
        # Check filepath exists with
        if not os.path.exists(markdown_filepath):
            if resource_exists('vectorhub', markdown_filepath):
                markdown_filepath = resource_filename('vectorhub', markdown_filepath)
            else:
                raise FileNotFoundError(f"Unable to find {markdown_filepath}.")
        if verbose:
            print(markdown_filepath)
        # Remove sys.argv, not sure what it was doing
        with open(markdown_filepath, encoding=encoding) as f:
            config = list(yaml.load_all(self._get_yaml(f), Loader=yaml.SafeLoader))
            text = f.read()
            self.config = config[0]
            for k,v in self.config.items():
                setattr(self, k, v)
            self.markdown_description = text
        self._split_markdown_description(text, splitter=splitter)

    def _split_markdown_description(self, description: str, splitter: str=r"(\#\#+\ +)|(\n)"):
        """
            Breaks markdown into heading and values.
            Args:
                description: Description of the markdown
                splitter: Regex to split the sentence. Currently it splits on headings and new lines.
                The purpose of this is to allow us to get keys from markdown files.
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
        for x in re.split(splitter, description):
            if x is None:
                continue
            if SKIP_NEW_LINE:
                if x == '\n':
                    continue

            if IS_HEADING:
                heading = x.lower().rstrip().replace(' ', '_')
                IS_HEADING = False
                # Skip new line after the heading is declared
                SKIP_NEW_LINE = True
                value = ""
            elif '##' in x:
                # Insert setting new layer
                if heading is not None:
                    setattr(self, heading, value)
                    markdown_values[heading] = value
                IS_HEADING = True
            else:
                SKIP_NEW_LINE = False
                value += x

        # Set the final value
        if hasattr(self, heading):
            if getattr(self, heading) != value:
                setattr(self, heading, value)
        else:
            setattr(self, heading, value)

