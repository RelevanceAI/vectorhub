---
# be sure to put quotations
model_id: "text/experts_bert"
model_name: "Experts BERT - Collection of BERT experts fine-tuned on different datasets."
vector_length: 768 (Bert Large)
paper: "https://arxiv.org/abs/1810.04805v2"
repo: "https://tfhub.dev/google/collections/experts/bert/1"
release_date: '2021-02-15'
installation: "pip install vectorhub[encoders-text-tfhub]"
category: text
---

## Description

Starting from a pre-trained BERT model and fine-tuning on the downstream task gives impressive results on many NLP tasks. One can further increase the performance by starting from a BERT model that better aligns or transfers to the task at hand, particularly when having a low number of downstream examples. For example, one can use a BERT model that was trained on text from a similar domain or by use a BERT model that was trained for a similar task.

This is a collection of such BERT "expert" models that were trained on a diversity of datasets and tasks to improve performance on downstream tasks like question answering, tasks that require natural language inference skills, NLP tasks in the medical text domain, and more.

![Bert Image](https://miro.medium.com/max/619/1*iJqlhZz-g6ZQJ53-rE9VvA.png)

## Working in Colab

If you are using this in colab and want to save this so you don't have to reload, use: 

```
import os 
os.environ['TFHUB_CACHE_DIR'] = "drive/MyDrive/"
os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "COMPRESSED"
```

## Example

```python
#pip install vectorhub[encoders-text-tfhub]
#FOR WINDOWS: pip install vectorhub[encoders-text-tfhub-windows]
from vectorhub.encoders.text.tfhub import ExpertsBert2Vec
model = ExpertsBert2Vec()
model.encode("I enjoy taking long walks along the beach with my dog.")
```

## Model Versions

Model Table | Vector Length 
------------| ---------- 
https://tfhub.dev/google/experts/bert/wiki_books/2 | 768  
https://tfhub.dev/google/experts/bert/wiki_books/mnli/2 | 768  
https://tfhub.dev/google/experts/bert/wiki_books/qnli/2 | 768  
https://tfhub.dev/google/experts/bert/wiki_books/qqp/2 | 768  
https://tfhub.dev/google/experts/bert/wiki_books/sst2/2 | 768  
https://tfhub.dev/google/experts/bert/wiki_books/squad2/2 | 768  
https://tfhub.dev/google/experts/bert/pubmed/2 | 768  
https://tfhub.dev/google/experts/bert/pubmed/squad2/2 | 768  


## Limitations

* NA

## Training Corpora:

* BooksCorpus (800M words)
* Wikipedia (2,500M words)
* MEDLINE/PubMed
* CORD-19
* CoLa
* MRPC

## Other Notes:

* NA


