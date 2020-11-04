"""
    Script to create model cards.
"""

if __name__=="__main__":
    from vectorai import ViClient
    from vectorai.models.deployed.text import ViText2Vec
    from vectorhub.auto_encoder import *
    import os

    vi_client = ViClient(os.environ['VH_USERNAME'], os.environ['VH_API_KEY'])
    text_encoder = ViText2Vec(os.environ['VH_USERNAME'], os.environ['VH_API_KEY'])
    docs =  get_model_definitions(None)
    vi_client.insert_documents(os.environ['VH_COLLECTION_NAME'], docs, models={'description':text_encoder})
