"""
    Script to create model cards. Uploads to VectorHub collection.
"""
import os
import argparse
import time
import re
import logging
from typing import List
# Wildcard import to get all classes

FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
LOGGER = logging.getLogger(__name__)
logging.basicConfig(format=FORMAT, level=logging.WARNING)
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.DEBUG)
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
LOGGER.addHandler(c_handler)

def remove_example_from_description(text):
    # Remove the Example if it is in the middle of the document
    text = re.sub(r'## Example(.*?)##', '##', text, flags=re.DOTALL)
    if '## Example' in text:
        # text = re.sub(r'## Example(.*)', '', text)
        text = re.sub(r"## Example(.*)\`\`\`.*?\`\`\`", '', text, flags=re.DOTALL, count=1)
    return text

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection_name', default='vh_markdown')
    parser.add_argument('--quick_run', action='store_true')
    parser.add_argument('--reset_collection', action='store_true')
    parser.add_argument('--evaluate_results', action='store_true')
    args = parser.parse_args()
    
    from vectorhub.auto_encoder import *
    from vectorai import ViClient
    from vectorai.models.deployed.text import ViText2Vec

    docs =  get_model_definitions(None)
    LOGGER.debug("Number of documents are: ")
    LOGGER.debug(len(docs))
    # Get _id across all documents

    LOGGER.debug("Marksdowns without example:")

    for i, doc in enumerate(docs):
        markdown_without_example = remove_example_from_description(doc['markdown_description'])
        docs[i]['markdown_without_example'] = markdown_without_example
        # LOGGER.debug(markdown_without_example)

    # Generate 1 sentence summaries for the models
    if not args.quick_run:
        from transformers import PegasusTokenizer, PegasusForConditionalGeneration
        mname = "google/pegasus-large"
        model = PegasusForConditionalGeneration.from_pretrained(mname)
        tok = PegasusTokenizer.from_pretrained(mname)

        def summarise(text):
            batch = tok.prepare_seq2seq_batch(src_texts=[text])  # don't need tgt_text for inference
            gen = model.generate(**batch)
            return tok.batch_decode(gen, skip_special_tokens=True)[0]

        for i, doc in enumerate(docs):
            if 'short_description' not in docs[i].keys():
                short_description = summarise(doc['description'])
                docs[i]['short_description'] = short_description
                # LOGGER.debug(short_description)

    vi_client = ViClient(os.environ['VH_USERNAME'], os.environ['VH_API_KEY'])
    ids = vi_client.get_field_across_documents('_id', docs)
    if args.reset_collection:
        if args.collection_name in vi_client.list_collections():
            vi_client.delete_collection(args.collection_name)
            time.sleep(5)
    text_encoder = ViText2Vec(os.environ['VH_USERNAME'], os.environ['VH_API_KEY'])

    response = vi_client.insert_documents(args.collection_name, docs, models={'description': text_encoder}, overwrite=True)

    LOGGER.debug(response)
    print(response)
    if response['failed'] != 0:
        raise ValueError("Failed IDs")
    
    if args.evaluate_results:
        LOGGER.debug("Checking Documents:")
        LOGGER.debug(vi_client.head(args.collection_name))
        LOGGER.debug(vi_client.head(args.collection_name)['vector_length'])
        LOGGER.debug(vi_client.collection_schema(args.collection_name))
        import pandas as pd
        pd.set_option('display.max_colwidth', None)
        LOGGER.debug(vi_client.show_json(vi_client.random_documents(args.collection_name), selected_fields=['markdown_without_example']))
