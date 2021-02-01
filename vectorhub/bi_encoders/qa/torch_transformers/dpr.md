---
model_id: "qa/dpr"
model_name: "Dense Passage Retrieval"
vector_length: "768 (default)"
release_date: "2020-10-04"
paper: "https://arxiv.org/abs/2004.04906"
installation: "pip install vectorhub[encoders-text-torch-transformers]"
category: question-answer
---

## Description

Open-domain question answering relies on efficient passage retrieval to select candidate contexts, where traditional sparse vector space models, such as TF-IDF or BM25, are the de facto method. In this work, we show that retrieval can be practically implemented using dense representations alone, where embeddings are learned from a small number of questions and passages by a simple dual-encoder framework. When evaluated on a wide range of open-domain QA datasets, our dense retriever outperforms a strong Lucene-BM25 system largely by 9%-19% absolute in terms of top-20 passage retrieval accuracy, and helps our end-to-end QA system establish new state-of-the-art on multiple open-domain QA benchmarks.

## Example

```
#pip install vectorhub[encoders-text-torch-transformers]
from vectorhub.bi_encoders.qa.torch_transformers import DPR2Vec
model = DPR2Vec()
model.encode_question('How is the weather today?')
model.encode_answer('The weather is great today.')
```
