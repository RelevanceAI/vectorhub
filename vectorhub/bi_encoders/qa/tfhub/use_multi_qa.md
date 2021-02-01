---
model_id: 'qa/use-multi-qa'
model_name: "Universal Sentence Encoder Multilingual Question Answering"
vector_length: "512 (default)"
repo: "https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3"
installation: "pip install vectorhub[encoders-text-tfhub]"
release_date: "2019-07-01"
category: question-answer
short_description: Greater-than-word length multi-lingual text encoder for question answer retrieval.
---

## Description

- Developed by researchers at Google, 2019, v2 [1].
- Covers 16 languages, strong performance on cross-lingual question answer retrieval.
- It is trained on a variety of data sources and tasks, with the goal of learning text representations that are useful out-of-the-box to retrieve an answer given a question, as well as question and answers across different languages.
- It can also be used in other applications, including any type of text classification, clustering, etc.


## Supported Languages

Arabic, Chinese-simplified, Chinese-traditional, English, French, German, Italian, Japanese, Korean, Dutch, Polish, Portuguese, Spanish, Thai, Turkish, Russian

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

```python
#pip install vectorhub[encoders-text-tfhub]
from vectorhub.bi_encoders.qa.tfhub import USEMultiQA2Vec
model = USEMultiQA2Vec()
model.encode_question('How is the weather today?')
model.encode_answer('The weather is great today.')
```
