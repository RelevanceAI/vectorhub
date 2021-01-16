"""
    Test Collection
"""
from vectorai import ViClient, ViCollectionClient

class TempClient:
    """Client For a temporary collection
    """
    def __init__(self, client, collection_name: str=None):
        if client is None: 
            raise ValueError("Client cannot be None.")
        self.client = client
        if isinstance(client, ViClient):
            self.collection_name = collection_name
        elif isinstance(client, ViCollectionClient):
            self.collection_name = self.client.collection_name
        else:
            self.collection_name = collection_name

    def teardown_collection(self):
        if self.collection_name in self.client.list_collections():
            time.sleep(2)
            if isinstance(self.client, ViClient):
                self.client.delete_collection(self.collection_name)
            elif isinstance(self.client, ViCollectionClient):
                self.client.delete_collection()
    
    def __enter__(self):
        self.teardown_collection()
        return self.client
    
    def __exit__(self, *exc):
        self.teardown_collection()
