# How to contribute to Vector AI?

Everyone is welcome to contribute, and we value everybody's contribution. Code
is thus not the only way to help the community. Answering questions, helping
others, reaching out and improving the documentations are immensely valuable to
the community.

It also helps us if you spread the word: reference the library from blog posts
on the awesome projects it made possible, shout out on Twitter/Reddit/Kaggle 
every time it has helped you, or simply star the repo to say "thank you".

# How To Add Your Model To Vector Hub

We have written a simple 7-step guide to help you add your models here if you have trained them! This should take approximately 30 minutes - 1 hour. Let us know at dev@vctr.ai if you need any help.

1. Fork the project.

2. Identify the minimum requirements for your model, identify the associated module and then add them to the MODEL_REQUIREMENTS in vectorhub/model_dict.

3. Write a brief description about what your model involves.

4. Create a new branch called new_model/____2vec, replace ___ with the model/domain etc.

5. Identify which directory your model should fall under. Here is a basic directory outline.

|____ encoders
|________ audio
|________ image
|________ text
|____ bi_encoders
|________ qa
If you believe your model falls under a new category than we recommend making a new directory!

6. Once you identify the requirements, find the associated module or create a new one if required.

Use the following code as a base for any new models and add an encode and bulk_encode method. Both should return lists.
```
    from ....import_utils import *
    # Import dictionary for model requirements
    from ....models_dict import MODEL_REQUIREMENTS
    # Add dependencies in if-statement to avoid import breaks in the library
    if is_all_dependency_installed(MODEL_REQUIREMENTS['text-bi-encoder-tfhub-use-qa']):
        # add imports here
        import bert
        import numpy as np
        import tensorflow as tf
        import tensorflow_hub as hub
        import tensorflow_text
    from typing import List
    # This decorator returns a default vector in case of an error
    from ....base import catch_vector_errors
    # Base class that provides basic utilities
    from ..base import BaseTextText2Vec

    class USEMultiQA2Vec(BaseTextText2Vec):
        ...
        # Add decorator in case encoding errors and we need a dummy vector.
        @catch_vector_errors
        def encode(self, text):
            pass
```
7. Submit a PR!