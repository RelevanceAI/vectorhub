---
model_id: 'text_text/use-qa'
model_name: "Universal Sentence Encoder Question Answering"
vector_length: "512 (default)"
release_date: "2020-03-11"
repo: 'https://tfhub.dev/google/universal-sentence-encoder-qa/3'
installation: "pip install vectorhub[encoders-text-tfhub-tftext]"
category: text-text
---

## Description

- Developed by researchers at Google, 2019, v2 [1].
- It is trained on a variety of data sources and tasks, with the goal of learning text representations that 
are useful out-of-the-box to retrieve an answer given a question, as well as question and answers across different languages.
- It can also be used in other applications, including any type of text classification, clustering, etc.

## Example

```
#pip install vectorhub[encoders-text-tfhub]
from vectorhub.bi_encoders.text_text.tfhub import USEQA2Vec
model = USEQA2Vec()
model.encode_question('How is the weather today?')
model.encode_answer('The weather is great today.')
```
