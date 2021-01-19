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
    
    @property
    def data_type(self):
        """
        Returns text/audio/image/qa
        """
        return self.model_id.split('/')[0]

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
                "release_date": self.release_date,
                "vectorai_integration": self.vectorai_integration
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

    @property
    def audio_items_examples(self):
        return [
            'https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_69.wav',
            'https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_99.wav',
            'https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_10.wav',
            'https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_5.wav'
        ]

    @property
    def audio_metadata_examples(self):
        ['male', 'male', 'female', 'male']

    @property
    def text_items_examples(self):
        return [
            "chicken",
            "toilet",
            "paper",
            "enjoy walking"
        ]
    
    @property
    def qa_items_examples(self):
        return [
            "A blue whale in North Atlantic can grow up to 90 feet.",
            "A blue whale in Antarctica can grow up to 110 feet.",
            "A gorilla can lift 4000pounds (1810kg) on a bench press.",
            "A well-trained man can lift up to 401.5kg."
        ]
    @property
    def qa_search_example(self):
        return "How long can a blue whale grow in Antarctica?"
    
    @property
    def qa_metadata_examples(self):
        return ["whale", "whale", "gorilla", "human"]

    @property
    def text_metadata_examples(self):
        return [
            {'num_of_letters': 7,
            'type': 'animal'},
            {'num_of_letters': 6,
            'type': 'household_items'},
            {'num_of_letters': 5,
            'type': 'household_items'},
            {'num_of_letters': 12,
            'type': 'emotion'}
        ]
    
    @property
    def image_items_examples(self):
        return [
            'https://getvectorai.com/_nuxt/img/rabbit.4a65d99.png',
            'https://getvectorai.com/_nuxt/img/dog-2.b8b4cef.png',
            'https://getvectorai.com/_nuxt/img/dog-1.3cc5fe1.png',
        ]

    @property
    def image_metadata_examples(self):
        return [
            {'animal': 'rabbit', 'hat': 'no'},
            {'animal': 'dog', 'hat': 'yes'},
            {'animal': 'dog', 'hat': 'yes'}
        ]

    @property
    def search_example(self):
        return self.DATA_TYPE_TO_EXAMPLE[self.data_type][2]

    @property
    def text_search_example(self):
        return 'basin'
    
    @property
    def image_search_example(self):
        return self.image_items_examples[2]
    
    @property
    def audio_search_example(self):
        return self.audio_items_examples[0]

    @property
    def item_examples(self):
        return self.DATA_TYPE_TO_EXAMPLE[self.data_type][0]

    @property
    def DATA_TYPE_TO_EXAMPLE(self):
        # Example items, example metadata, example search
        return {
            'text': (self.text_items_examples, self.text_metadata_examples, self.text_search_example),
            'image': (self.image_items_examples, self.image_metadata_examples, self.image_search_example),
            'audio': (self.audio_items_examples, self.audio_metadata_examples, self.audio_search_example),
            'qa': (self.qa_items_examples, self.qa_metadata_examples, self.qa_search_example)
        }

    @property
    def metadata_examples(self):
        return self.DATA_TYPE_TO_EXAMPLE[self.data_type][1]

    @property
    def vectorai_integration(self):
        return f"""Index and search your vectors easily on the cloud using 1 line of code!

```
username = '<your username>'
email = '<your email>'
# You can request an api_key using - type in your username and email.
api_key = model.request_api_key(username, email)

# Index in 1 line of code
items = {self.item_examples}
model.add_documents(user, api_key, items)

# Search in 1 line of code and get the most similar results.
model.search('{self.search_example}')

# Add metadata to your search
metadata = {self.metadata_examples}
model.add_documents(user, api_key, items, metadata=metadata)
```
        """

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

