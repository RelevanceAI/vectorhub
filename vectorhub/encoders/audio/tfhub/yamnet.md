---
model_id: audio/yamnet
model_name: Yamnet 
vector_length: "1024 (default)"
release_date: "2020-03-11"
repo: "https://tfhub.dev/google/yamnet/1"
installation: "pip install vectorhub[encoders-audio-tfhub]"
category: audio
short_description: YAMNet is a fast and accurate audio event classifier that can be used for a variety of audio tasks.
---

## Description

YAMNet is an audio event classifier that takes audio waveform as input and makes independent predictions for each 
of 521 audio events from the AudioSet ontology. The model uses the MobileNet v1 architecture and was trained using 
the AudioSet corpus. This model was originally released in the TensorFlow Model Garden, where we have the model 
source code, the original model checkpoint, and more detailed documentation.
This model can be used: 

- as a stand-alone audio event classifier that provides a reasonable baseline across a wide variety of audio events.
- as a high-level feature extractor: the 1024-D embedding output of YAMNet can be used as the input features of another shallow model which can then be trained on a small amount of data for a particular task. This allows quickly creating specialized audio classifiers without requiring a lot of labeled data and without having to train a large model end-to-end.
- as a warm start: the YAMNet model parameters can be used to initialize part of a larger model which allows faster fine-tuning and model exploration.

## Working in Colab

If you are using this in colab and want to save this so you don't have to reload, use: 

```
import os 
os.environ['TFHUB_CACHE_DIR'] = "drive/MyDrive/"
os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "COMPRESSED"
```

## Example

```
#pip install vectorhub[encoders-audio-tfhub]
from vectorhub.encoders.audio.tfhub import Yamnet2Vec
model = Yamnet2Vec()
sample = model.read('https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_2.wav')
model.encode(sample)
```

## Limitations

YAMNet's classifier outputs have not been calibrated across classes, so you cannot directly treat 
the outputs as probabilities. For any given task, you will very likely need to perform a calibration with task-specific data 
which lets you assign proper per-class score thresholds and scaling.
YAMNet has been trained on millions of YouTube videos and although these are very diverse, there can still be a domain mismatch 
between the average YouTube video and the audio inputs expected for any given task. You should expect to do some amount of 
fine-tuning and calibration to make YAMNet usable in any system that you build.
