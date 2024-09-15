from pymongo import MongoClient
from pymongo.collection import Collection
from bson.objectid import ObjectId

class DocumentRepository:
    def __init__(self, db_name='GPT-VOICE-ASSISTANT', collection_name='doc-ingestion-config'):
        self.client = MongoClient('mongodb://localhost:27017/')
        self.db = self.client[db_name]
        self.collection: Collection = self.db[collection_name]

    def create_document(self, storage_index: str, name: str, ingested_files: list):
        document = {
            "storageIndex": storage_index,
            "name": name,
            "ingestedFiles": ingested_files
        }
        result = self.collection.insert_one(document)
        return result.inserted_id

    def get_document(self, storage_index: str):
        return self.collection.find_one({"storageIndex": storage_index})

    def update_document(self, storage_index: str, update_fields: dict):
        self.collection.update_one(
            {"storageIndex": storage_index},
            {"$set": update_fields}
        )
    def get_collection_names_with_storage_index(self):
        # Get the list of collections in the database

        # Create a list to hold the results
        result = []

        # Loop through each collection and retrieve storage index and name
        collection= self.collection
            
        # Find the first document in the collection to extract the storage index and name
        documents = collection.find({}, {"storageIndex": 1, "name": 1})

            # Loop through all documents in the collection
        for document in documents:
            result.append({
                "storageIndex": document.get("storageIndex", None),
                "name": document.get("name", None)
            })

        return result
