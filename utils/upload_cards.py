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
    vi_client = ViClient(os.environ['VH_USERNAME'], os.environ['VH_API_KEY'])
    if args.collection_name in vi_client.list_collections():
        vi_client.delete_collection(args.collection_name)
        time.sleep(5)
    text_encoder = ViText2Vec(os.environ['VH_USERNAME'], os.environ['VH_API_KEY'])
    vi_client.insert_documents(args.collection_name, docs)
    print("Checking Documents:")
    print(vi_client.head('vh_markdown_2'))
    print(vi_client.collection_schema('vh_markdown_2'))
