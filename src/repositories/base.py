"""
Generic Repository Base Class
DRY foundation for async CRUD operations on MongoDB collections.
"""
from typing import Generic, TypeVar, Type, Optional, List, Dict, Any
from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase
from bson import ObjectId
import datetime as dt

from ..models.base import MongoBaseModel
from ..utils.observability import logger

# Generic type for domain models    
T = TypeVar("T", bound=MongoBaseModel)


class BaseRepository(Generic[T]):
    """
    Generic async repository for MongoDB collections.
    Provides type-safe CRUD operations for domain models.

    Usage:
        class LeadRepository(BaseRepository[Lead]):
            def __init__(self, database: AsyncIOMotorDatabase):
                super().__init__(database, "leads", Lead)
    """

    def __init__(
        self,
        database: AsyncIOMotorDatabase,
        collection_name: str,
        model_class: Type[T]
    ):
        """
        Initialize repository with database connection and model type.

        Args:
            database: Motor database instance
            collection_name: MongoDB collection name
            model_class: Pydantic model class for type safety
        """
        self.database = database
        self.collection: AsyncIOMotorCollection = database[collection_name]
        self.model_class = model_class
        self.collection_name = collection_name

    async def create(self, document: T) -> T:
        """
        Insert a new document into the collection.

        Args:
            document: Domain model instance to persist

        Returns:
            The created document with `_id` populated

        Raises:
            pymongo.errors.DuplicateKeyError: If unique constraint violated
        """
        # Update timestamps
        now = dt.datetime.now(dt.UTC)
        document.created_at = now
        document.updated_at = now

        # Convert Pydantic model to dict for MongoDB
        doc_dict = document.model_dump(
            by_alias=True, 
            exclude={"id"},
            exclude_none=True)

        for computed_field in type(document).model_computed_fields:
            doc_dict.pop(computed_field, None)

        result = await self.collection.insert_one(doc_dict)

        logger.debug(
            f"Created document in {self.collection_name}",
            extra={"document_id": str(result.inserted_id)}
        )

        # Populate the _id field
        document.id = str(result.inserted_id)
        return document

    async def find_by_id(self, document_id: str) -> Optional[T]:
        """
        Retrieve a document by its MongoDB ObjectId.

        Args:
            document_id: String representation of ObjectId

        Returns:
            Domain model instance or None if not found
        """
        doc = await self.collection.find_one({"_id": ObjectId(document_id)})

        if doc is None:
            return None

        return self._to_model(doc)

    async def find_one(self, filter_dict: Dict[str, Any]) -> Optional[T]:
        """
        Retrieve the first document matching the filter.

        Args:
            filter_dict: MongoDB query filter

        Returns:
            Domain model instance or None if not found
        """
        doc = await self.collection.find_one(filter_dict)

        if doc is None:
            return None

        return self._to_model(doc)

    async def find_many(
        self,
        filter_dict: Dict[str, Any],
        limit: int = 100,
        skip: int = 0,
        sort: Optional[List[tuple]] = None
    ) -> List[T]:
        """
        Retrieve multiple documents matching the filter.

        Args:
            filter_dict: MongoDB query filter
            limit: Maximum number of documents to return
            skip: Number of documents to skip (pagination)
            sort: List of (field, direction) tuples for sorting

        Returns:
            List of domain model instances
        """
        cursor = self.collection.find(filter_dict).skip(skip).limit(limit)

        if sort:
            cursor = cursor.sort(sort)

        docs = await cursor.to_list(length=limit)

        return [self._to_model(doc) for doc in docs]

    async def update(self, document: T) -> T:
        """
        Update an existing document by its _id.

        Args:
            document: Domain model instance with populated `id` field

        Returns:
            Updated domain model instance

        Raises:
            ValueError: If document has no `id` field
            RuntimeError: If document not found
        """
        if not document.id:
            raise ValueError("Cannot update document without an id")

        # Update timestamp
        document.updated_at = dt.datetime.now(dt.UTC)

        # Convert to dict, excluding the _id field from the update
        doc_dict = document.model_dump(by_alias=True, exclude={"id"})

        result = await self.collection.update_one(
            {"_id": ObjectId(document.id)},
            {"$set": doc_dict}
        )

        if result.matched_count == 0:
            raise RuntimeError(f"Document with id {document.id} not found")

        logger.debug(
            f"Updated document in {self.collection_name}",
            extra={"document_id": document.id}
        )

        return document

    async def delete(self, document_id: str) -> bool:
        """
        Delete a document by its MongoDB ObjectId.

        Args:
            document_id: String representation of ObjectId

        Returns:
            True if document was deleted, False if not found
        """
        result = await self.collection.delete_one({"_id": ObjectId(document_id)})

        if result.deleted_count > 0:
            logger.debug(
                f"Deleted document from {self.collection_name}",
                extra={"document_id": document_id}
            )
            return True

        return False

    async def count(self, filter_dict: Optional[Dict[str, Any]] = None) -> int:
        """
        Count documents matching the filter.

        Args:
            filter_dict: MongoDB query filter (None for all documents)

        Returns:
            Number of matching documents
        """
        filter_dict = filter_dict or {}
        return await self.collection.count_documents(filter_dict)

    async def bulk_create(self, documents: List[T]) -> List[T]:
        """
        Insert multiple documents in a single operation.

        Args:
            documents: List of domain model instances

        Returns:
            List of created documents with `_id` populated
        """
        if not documents:
            return []

        now = dt.datetime.now(dt.UTC)

        # Convert to dicts and update timestamps
        doc_dicts = []
        for doc in documents:
            doc.created_at = now
            doc.updated_at = now
            doc_dicts.append(doc.model_dump(by_alias=True, exclude={"id"}))

        result = await self.collection.insert_many(doc_dicts)

        # Populate _id fields
        for doc, inserted_id in zip(documents, result.inserted_ids):
            doc.id = str(inserted_id)

        logger.debug(
            f"Bulk created documents in {self.collection_name}",
            extra={"count": len(documents)}
        )

        return documents

    def _to_model(self, doc: Dict[str, Any]) -> T:
        """
        Convert MongoDB document to Pydantic model instance.

        Args:
            doc: Raw MongoDB document dict

        Returns:
            Domain model instance
        """
        if not doc:
            return None
        # Convert ObjectId to string for Pydantic validation
        if "_id" in doc:
            doc["_id"] = str(doc["_id"])

        model_fields = self.model_class.model_fields.keys()

        cleaned_doc = {
            k: v for k, v in doc.items() 
            if k in model_fields or k == "_id"
        }

        return self.model_class.model_validate(cleaned_doc)
