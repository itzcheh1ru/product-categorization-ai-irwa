from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class MongoDBService:
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        self._available = False
        # Fallback in-memory store when MongoDB is unavailable
        self._memory_store: list[Dict[str, Any]] = []
        self._connect()
    
    def _connect(self):
        """Connect to MongoDB"""
        try:
            # MongoDB connection string
            connection_string = "mongodb+srv://amayagunawardhana6_db_user:LF5Q3uuxenNhWTPY@productdb.d3zxyx0.mongodb.net/"
            database_name = "ProductDB"
            collection_name = "products"
            
            # Create both sync and async clients with very short selection timeout to avoid long startup delays
            self.client = MongoClient(connection_string, serverSelectionTimeoutMS=1000, connectTimeoutMS=1000)
            self.async_client = AsyncIOMotorClient(connection_string, serverSelectionTimeoutMS=1000, connectTimeoutMS=1000)
            
            # Get database and collection
            self.db = self.client[database_name]
            self.async_db = self.async_client[database_name]
            self.collection = self.db[collection_name]
            self.async_collection = self.async_db[collection_name]
            
            # Test connection
            self.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
            self._available = True
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            # Do not raise on startup; fall back to in-memory store so API can run
            self._available = False
            self.client = None
            self.db = None
            self.collection = None
    
    def add_product(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add a new product to MongoDB"""
        try:
            # Add metadata
            product_data['created_at'] = datetime.utcnow()
            product_data['updated_at'] = datetime.utcnow()
            
            if not self._available or self.collection is None:
                # In-memory fallback
                product_data['id'] = str(len(self._memory_store) + 1)
                self._memory_store.append(product_data.copy())
                inserted_product = product_data
                result_id = product_data['id']
            else:
                # Insert product
                result = self.collection.insert_one(product_data)
                # Get the inserted product
                inserted_product = self.collection.find_one({"_id": result.inserted_id})
                # Convert ObjectId to string for JSON serialization
                if inserted_product:
                    inserted_product['_id'] = str(inserted_product['_id'])
                    inserted_product['id'] = str(result.inserted_id)  # Add id field for compatibility
                result_id = str(result.inserted_id)
            
            return {
                "success": True,
                "product_id": result_id,
                "product": inserted_product
            }
            
        except Exception as e:
            logger.error(f"Error adding product to MongoDB: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_products(self, limit: int = 100, skip: int = 0) -> List[Dict[str, Any]]:
        """Get products from MongoDB"""
        try:
            if not self._available or self.collection is None:
                products = list(self._memory_store)[skip: skip + limit]
            else:
                products = list(self.collection.find().skip(skip).limit(limit).sort("created_at", -1))
            
            # Convert ObjectIds to strings
            for product in products:
                if '_id' in product:
                    product['_id'] = str(product['_id'])
                    product['id'] = str(product['_id'])
            
            return products
            
        except Exception as e:
            logger.error(f"Error getting products from MongoDB: {e}")
            return []
    
    def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific product by ID"""
        try:
            if not self._available or self.collection is None:
                product = next((p for p in self._memory_store if p.get('id') == product_id), None)
            else:
                from bson import ObjectId
                product = self.collection.find_one({"_id": ObjectId(product_id)})
            
            if product:
                if '_id' in product:
                    product['_id'] = str(product['_id'])
                    product['id'] = str(product['_id'])
            
            return product
            
        except Exception as e:
            logger.error(f"Error getting product by ID: {e}")
            return None
    
    def search_products(self, query: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """Search products based on criteria"""
        try:
            if not self._available or self.collection is None:
                def matches(doc: Dict[str, Any]) -> bool:
                    for k, v in query.items():
                        if doc.get(k) != v:
                            return False
                    return True
                products = [p for p in self._memory_store if matches(p)][:limit]
            else:
                products = list(self.collection.find(query).limit(limit).sort("created_at", -1))
            
            # Convert ObjectIds to strings
            for product in products:
                if '_id' in product:
                    product['_id'] = str(product['_id'])
                    product['id'] = str(product['_id'])
            
            return products
            
        except Exception as e:
            logger.error(f"Error searching products: {e}")
            return []
    
    def update_product(self, product_id: str, update_data: Dict[str, Any]) -> bool:
        """Update a product in MongoDB"""
        try:
            update_data['updated_at'] = datetime.utcnow()
            if not self._available or self.collection is None:
                for i, p in enumerate(self._memory_store):
                    if p.get('id') == product_id:
                        self._memory_store[i] = {**p, **update_data}
                        return True
                return False
            else:
                from bson import ObjectId
                result = self.collection.update_one(
                    {"_id": ObjectId(product_id)},
                    {"$set": update_data}
                )
                return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error updating product: {e}")
            return False
    
    def delete_product(self, product_id: str) -> bool:
        """Delete a product from MongoDB"""
        try:
            if not self._available or self.collection is None:
                before = len(self._memory_store)
                self._memory_store = [p for p in self._memory_store if p.get('id') != product_id]
                return len(self._memory_store) < before
            else:
                from bson import ObjectId
                result = self.collection.delete_one({"_id": ObjectId(product_id)})
                return result.deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error deleting product: {e}")
            return False
    
    def get_products_count(self) -> int:
        """Get total number of products"""
        try:
            if not self._available or self.collection is None:
                return len(self._memory_store)
            return self.collection.count_documents({})
        except Exception as e:
            logger.error(f"Error getting products count: {e}")
            return 0

# Global instance (safe even if MongoDB is unavailable)
mongodb_service = MongoDBService()
