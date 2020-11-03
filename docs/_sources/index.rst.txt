
Welcome to VectorHub's documentation!
=====================================

Vector Hub is your home for ___2Vec models!



The rise of deep learning and encoding has meant that there are now explosion of
open-source and proprietary models and techniques that have allowed for distributed
representation of entities. This means the rise of new ____2Vec models that are:

1) Model-specific - New architecture is introduced.
2) Domain-specific - Architecture is trained on new domain.
3) Language-specific - Architecture is trained in new language.
4) Task-specific - Architecture is trained on new task.

In order to allow people to understand what these models do and mean, we aim to provide 
a hub for these __2vec models.

Our vision to build a hub that allows people to store these ____2Vec models and provide explanations
for how to best use these encodings while building a flexible framework that allows these 
different models to be used easily.




.. toctree::
   :maxdepth: 2
   :caption: Contents

   intro
   how_to_add_a_model
   auto_encoder

.. toctree::
   :maxdepth: 4
   :caption: Text Encoders

   encoders.text.bert2vec
   encoders.text.albert2vec
   encoders.text.labse2vec
   encoders.text.use2vec
   encoders.text.use_multi2vec
   encoders.text.legalbert2vec
   encoders.text.transformer2vec
   encoders.text.sentencetransformer2vec
   encoders.text.vectorai2vec
   

.. toctree::
   :maxdepth: 2
   :caption: Image Encoders

   encoders.image.bit2vec
   encoders.image.inception2vec
   encoders.image.resnet2vec
   encoders.image.inception_resnet2vec
   encoders.image.mobilenet2vec
   encoders.image.vectorai2vec

.. toctree::
   :maxdepth: 2
   :caption: Audio Encoders

   encoders.audio.speech_embedding2vec
   encoders.audio.trill2vec
   encoders.audio.vggish2vec
   encoders.audio.yamnet2vec
   encoders.audio.wav2vec
   encoders.audio.vectorai2vec


.. toctree::
   :maxdepth: 4
   :caption: Text Bi-Encoders

   bi_encoders.text_text.use_qa2vec
   bi_encoders.text_text.lareqa_qa2vec
   bi_encoders.text_text.dpr2vec

.. toctree::
   :maxdepth: 4
   :caption: Modules

   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
