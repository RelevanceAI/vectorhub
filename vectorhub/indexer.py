"""
Indexer with the model.
"""
import warnings
from vectorai import ViClient, request_api_key
from typing import List, Any, Optional

class ViIndexer:
    @property
    def encoder_type(self):
        """The encoder type ensures it uses either the 'encode' or 'encode_question'/'encode_answer'
        Currently supported encoder types: 
            Question-Answer
            Encoder
        """
        if self.definition.model_id.startswith('qa'):
            return 'qa'
        else:
            return 'encoder'

    def request_api_key(self, username: str, email: str, referral=None):
        """
        Requesting an API key.
        """
        print("API key is being requested. Be sure to save it somewhere!")
        return request_api_key(username=username, email=email,
                        description='vectorhub', referral=referral)

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
            docs = [self._create_document(i, item) for i, item in enumerate(items)]

        self.client = ViClient(username, api_key)
        if self.encoder_type == 'encoder':
            return self.client.insert_documents(self.collection_name, docs, {'item': self})
        elif self.encoder_type == 'qa':
            return self.client.insert_documents(self.collection_name, docs, {'item': self.encode_question})

    def _create_document(self, _id: str, item: List[str], metadata=None):
        return {
            '_id': str(_id),
            'item': item,
            'metadata': metadata
        }
    
    def delete_collection(self, collection_name=None):
        if collection_name is None:
            collection_name = self.collection_name
        return self.delete_collection(collection_name)

    def search(self, item: Any, num_results: int=10):
        """
        Simple search with Vector AI
        """
        warnings.warn("If you are looking for more advanced functionality, we recommend using the official Vector AI Github package")
        if self.encoder_type == 'encoder':
            return self.client.search(self.collection_name, self.encode(item), 
            field='item_' + self.__name__ + '_vector_', page_size=num_results)
        elif self.encoder_type == 'qa':
            return self.client.search(self.collection_name, self.encode_question(item), 
            field='item_vector_', page_size=num_results)

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
