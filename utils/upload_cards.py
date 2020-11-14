"""
    Script to create model cards. Uploads to VectorHub collection.
"""

if __name__=="__main__":
    import os
    import argparse
    import time
    import re
    from vectorai import ViClient
    from vectorai.models.deployed.text import ViText2Vec
    from transformers import PegasusTokenizer, PegasusForConditionalGeneration
    from typing import List
    # Wildcard import to get all classes
    from vectorhub.auto_encoder import *

    parser = argparse.ArgumentParser()
    parser.add_argument('--collection_name', default=os.environ['VH_COLLECTION_NAME'])
    parser.add_argument('--quick_run', action='store_true')
    parser.add_argument('--reset_collection', action='store_true')
    parser.add_argument('--evaluate_results', action='store_true')
    args = parser.parse_args()

    docs =  get_model_definitions(None)
    print("Number of documents are: ")
    print(len(docs))
    print("Marksdowns without example:")
    def remove_example_from_description(text):
        # Remove the Example if it is in the middle of the document
        text = re.sub(r'## Example(.*?)##', '##', text, flags=re.DOTALL)
        if '## Example' in text:
            text = re.sub(r'## Example(.*)', '', text)
            text = re.sub(r"\`\`\`.*?\`\`\`", '', text, flags=re.DOTALL)
            # Remove if it is at the bottom of the document
        return text

    for i, doc in enumerate(docs):
        markdown_without_example = remove_example_from_description(doc['markdown_description'])
        docs[i]['markdown_without_example'] = markdown_without_example
        print(markdown_without_example)

    # Generate 1 sentence summaries for the models
    mname = "google/pegasus-large"

    model = PegasusForConditionalGeneration.from_pretrained(mname)
    tok = PegasusTokenizer.from_pretrained(mname)

    def summarise(text):
        batch = tok.prepare_seq2seq_batch(src_texts=[text])  # don't need tgt_text for inference
        gen = model.generate(**batch)
        return tok.batch_decode(gen, skip_special_tokens=True)[0]

    if not args.quick_run:
        for i, doc in enumerate(docs):
            short_description = summarise(doc['description'])
            docs[i]['short_description'] = short_description
            print(short_description)

    vi_client = ViClient(os.environ['VH_USERNAME'], os.environ['VH_API_KEY'])
    if args.reset_collection:
        if args.collection_name in vi_client.list_collections():
            vi_client.delete_collection(args.collection_name)
            time.sleep(5)
    text_encoder = ViText2Vec(os.environ['VH_USERNAME'], os.environ['VH_API_KEY'])
    response = vi_client.insert_documents(args.collection_name, docs, models={'description': text_encoder})
    if response['failed'] != 0:
        print(response)
        raise ValueError("Failed IDs")
    
    if args.evaluate_results:
        print("Checking Documents:")
        print(vi_client.head(args.collection_name))
        print(vi_client.head(args.collection_name)['vector_length'])
        print(vi_client.collection_schema(args.collection_name))
        import pandas as pd
        pd.set_option('display.max_colwidth', None)
        print(vi_client.show_json(vi_client.random_documents(args.collection_name), selected_fields=['markdown_without_example']))
