"""
Indexer with the model.
"""
import warnings
from vectorai import ViClient, request_api_key
from typing import List, Any, Optional

class ViIndexer:
    def request_api_key(self, username: str, email: str, referral=None):
        """
        Requesting an API key.
        """
        print("API key is being requested. Be sure to save it somewhere!")
        return request_api_key(username=username, email=email,
                        reason='vectorhub', referral=referral)

    def add_documents(self, username: str, api_key: str,
            items: List[Any], metadata: Optional[List[Any]]=None,
            collection_name: str=None):
        """
        Add documents to the Vector AI cloud.
        """
        self.username = username
        self.api_key = api_key
        if collection_name is not None:
            self.collection_name = collection_name
        else:
            self.collection_name = 'vectorhub_collection_with_' + self.__name__
        if metadata is not None:
            docs = [self._create_document(i, item, metadata) for i, (item, metadata) in enumerate(list(zip(items, metadata)))]
        else:
            docs = [self._create_document(i, item) for i, (item, metadata) in enumerate(items)]

        self.client = ViClient(username, api_key)
        return self.client.insert_documents(collection_name, docs, {'item': self})

    def _create_document(self, _id: str, item: List[str], metadata=None):
        return {
            '_id': str(_id),
            'item': item,
            'metadata': metadata
        }

    def search(self, item: Any, num_results: int=10):
        """
        Simple search with Vector AI
        """
        warnings.warn("If you are looking for more advanced functionality, we recommend using the official Vector AI Github package")
        return self.client.search(collection_name, self.encode(item), field='item_' + self.__name__ + '_vector_', page_size=num_results)

    def retrieve_documents(self, num_of_documents: int):
        """
        Get all the documents in our package.
        """
        return self.client.retrieve_documents(self.collection_name, page_size=num_of_documents)['documents']

    def retrieve_all_documents(self):
        """
        Retrieve all documents.
        """
        return self.retrieve_all_documents(self.collection_name)


