---
model_id: 'qa/use-qa'
model_name: "Universal Sentence Encoder Question Answering"
vector_length: "512 (default)"
release_date: "2020-03-11"
repo: 'https://tfhub.dev/google/universal-sentence-encoder-qa/3'
installation: "pip install vectorhub[encoders-text-tfhub-tftext]"
category: question-answer
short_description: Greater-than-word length text encoder for question answer retrieval.
---

## Description

- Developed by researchers at Google, 2019, v2 [1].
- It is trained on a variety of data sources and tasks, with the goal of learning text representations that 
are useful out-of-the-box to retrieve an answer given a question, as well as question and answers across different languages.
- It can also be used in other applications, including any type of text classification, clustering, etc.
- Multi-task training setup is based on the paper [Learning Cross-lingual Sentence Representations via a Multi-task Dual Encoder](https://arxiv.org/pdf/1810.12836.pdf)
- Achieved 56.1 on dev set in Squad Retrieval and 46.2 on train.

## Training Corpora

Reddit, Wikipedia, Stanford Natural Language Inference and web mined translation pairs.

## Training Setup 

Question-Answering was trained on 4 unique task types:
i) conversational response prediction
ii) quick thought 
iii) natural language inference 
iv) tranlsation ranking (bridge task)

Note: to learn cross-lingual representations, they used translation ranking tasks using parallel corpora for the source-target pairs. 

Multi-task training is performed through different tasks and performed an optimization step for a single task at a time. 
All models are trained with a batch size of 100 using SGD with a learning rate of 0.008 and 30million steps.

## Example

```
#pip install vectorhub[encoders-text-tfhub]
from vectorhub.bi_encoders.qa.tfhub import USEQA2Vec
model = USEQA2Vec()
model.encode_question('How is the weather today?')
model.encode_answer('The weather is great today.')
```
