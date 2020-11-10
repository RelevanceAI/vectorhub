"""
    Script to create model cards. Uploads to VectorHub collection.
"""

if __name__=="__main__":
    from vectorai import ViClient
    from vectorai.models.deployed.text import ViText2Vec
    from vectorhub.auto_encoder import *
    import os
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--collection_name', default=os.environ['VH_COLLECTION_NAME'])
    args = parser.parse_args()
    
    docs =  get_model_definitions(None)

    # Generate 1 sentence summaries for the models
    from transformers import PegasusTokenizer, PegasusForConditionalGeneration
    from typing import List
    mname = "google/pegasus-xsum"

    model = PegasusForConditionalGeneration.from_pretrained(mname)
    tok = PegasusTokenizer.from_pretrained(mname)
    # batch = tok.prepare_seq2seq_batch(src_texts=[PGE_ARTICLE])  # don't need tgt_text for inference
    # gen = model.generate(**batch)  # for forward pass: model(**batch)
    # summary: List[str] = tok.batch_decode(gen, skip_special_tokens=True)

    def summarise(text):
        batch = tok.prepare_seq2seq_batch(src_texts=[text])  # don't need tgt_text for inference
        gen = model.generate(**batch)
        return tok.batch_decode(gen, skip_special_tokens=True)[0]
    
    for i, doc in enumerate(docs):
        short_description = summarise(doc['description'])
        docs[i]['short_description'] = short_description

    vi_client = ViClient(os.environ['VH_USERNAME'], os.environ['VH_API_KEY'])
    if args.collection_name in vi_client.list_collections():
        vi_client.delete_collection(args.collection_name)
        time.sleep(5)
    text_encoder = ViText2Vec(os.environ['VH_USERNAME'], os.environ['VH_API_KEY'])
    vi_client.insert_documents(args.collection_name, docs, models={'description': text_encoder})
    print("Checking Documents:")
    print(vi_client.head(args.collection_name))
    print(vi_client.head(args.collection_name)['vector_length'])
    print(vi_client.collection_schema(args.collection_name))
