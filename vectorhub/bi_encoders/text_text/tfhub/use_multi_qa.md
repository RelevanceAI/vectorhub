---
model_id: 'text_text/use-multi-qa'
model_name: "Universal Sentence Encoder Multilingual Question Answering"
vector_length: "512 (default)"
repo: "https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3"
installation: "pip install vectorhub[encoders-text-tfhub]"
release_date: "2019-07-01"
category: text-text
---

## Description

- Developed by researchers at Google, 2019, v2 [1].
- Covers 16 languages, strong performance on cross-lingual question answer retrieval.
- It is trained on a variety of data sources and tasks, with the goal of learning text representations that are useful out-of-the-box to retrieve an answer given a question, as well as question and answers across different languages.
- It can also be used in other applications, including any type of text classification, clustering, etc.

## Example

```python
#pip install vectorhub[encoders-text-tfhub]
from vectorhub.bi_encoders.text_text.tfhub import USEMultiQA2Vec
model = USEMultiQA2Vec()
model.encode_question('How is the weather today?')
model.encode_answer('The weather is great today.')
```
