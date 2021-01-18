---
model_id: 'qa/lareqa-qa'
model_name: "LAReQA: Language-agnostic answer retrieval from a multilingual pool"
vector_length: '512 (default)'
paper: "https://arxiv.org/abs/2004.05484"
repo: "https://tfhub.dev/google/LAReQA/mBERT_En_En/1"
release_date: "2020-04-11"
installation: "pip install vectorhub[encoders-text-tfhub]"
category: text-text
---

## Description

We present LAReQA, a challenging new benchmark for language-agnostic answer retrieval from a multilingual candidate pool. Unlike previous cross-lingual tasks, LAReQA tests for "strong" cross-lingual alignment, requiring semantically related cross-language pairs to be closer in representation space than unrelated same-language pairs. Building on multilingual BERT (mBERT), we study different strategies for achieving strong alignment. We find that augmenting training data via machine translation is effective, and improves significantly over using mBERT out-of-the-box. Interestingly, the embedding baseline that performs the best on LAReQA falls short of competing baselines on zero-shot variants of our task that only target "weak" alignment. This finding underscores our claim that languageagnostic retrieval is a substantively new kind of cross-lingual evaluation.

## Example

```
#pip install vectorhub[encoders-text-tfhub]
from vectorhub.bi_encoders.qa.tfhub import LAReQA2Vec
model = LAReQA2Vec()
model.encode_question('How is the weather today?')
model.encode_answer('The weather is great today.')
```
