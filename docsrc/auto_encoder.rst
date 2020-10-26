Guide to using Auto-Encoder
=====================================

Inspired by transformers' adoption of the auto-models, we created an 
AutoEncoder class that allows you to easily get the relevant models. Not to be confused with the autoencoder architecture.

The relevant models can be found here: 

.. code-block:: python

    from vectorhub import AutoEncoder
    encoder = AutoEncoder('text/bert')
    encoder.encode("Hi...")


To view the list of available models, you can call: 


.. code-block:: python

    import vectorhub as vh 
    vh.list_available_auto_models()

When you instantiate the autoencoder, you will need to pip install 
the relevant module. The requirements here can be given here.

The list of supported models are:

.. code-block:: python

    ['text/albert', 'text/bert', 'text/labse', 'text/use', 'text/use-multi', 'text/use-lite', 'audio/fairseq', 'audio/speech_embedding', 'audio/trill', 'audio/trill-distilled', 'audio/vggish', 'audio/yamnet', 'image/bit', 'image/bit-medium', 'image/inception', 'image/inception-v2', 'image/inception-v3', 'image/inception-resnet', 'image/mobilenet', 'image/mobilenet-v2', 'image/resnet', 'image/resnet-v2', 'text_text/use-multiqa', 'text_text/lareqa-qa', 'text_text/dpr']
