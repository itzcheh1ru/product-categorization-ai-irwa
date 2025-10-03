"""
MongoDB Connection Module for Product Categorization AI
Handles database connection, data migration, and CRUD operations
"""

import pandas as pd
from pymongo import MongoClient
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class MongoDBManager:
    """MongoDB connection and data management class"""
    
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        self.connection_string = "mongodb+srv://amayagunawardhana6_db_user:LF5Q3uuxenNhWTPY@productdb.d3zxyx0.mongodb.net/"
        self.database_name = "ProductDB"
        self.collection_name = "products"
        
    def connect(self) -> bool:
        """Connect to MongoDB Atlas"""
        try:
            # Add SSL configuration to handle certificate issues
            self.client = MongoClient(
                self.connection_string,
                tls=True,
                tlsAllowInvalidCertificates=True,
                serverSelectionTimeoutMS=5000
            )
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
            
            # Test connection
            self.client.admin.command('ping')
            logger.info(f"Successfully connected to MongoDB Atlas - Database: {self.database_name}")
            # Ensure indexes exist for fast search
            try:
                self.create_text_index()
            except Exception:
                pass
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    def migrate_csv_to_mongodb(self, csv_file_path: str) -> bool:
        """Migrate CSV data to MongoDB"""
        try:
            # Read CSV file
            df = pd.read_csv(csv_file_path)
            logger.info(f"Loaded CSV file with {len(df)} records")
            
            # Clean the data - handle NaN values
            df = df.fillna('')  # Replace NaN with empty string
            
            # Convert DataFrame to list of dictionaries
            products = df.to_dict('records')
            
            # Add metadata and clean data
            for product in products:
                product['created_at'] = datetime.utcnow()
                product['updated_at'] = datetime.utcnow()
                product['source'] = 'csv_migration'
                
                # Convert any remaining NaN values to None
                for key, value in product.items():
                    if pd.isna(value):
                        product[key] = None
            
            # Clear existing collection (optional - remove if you want to keep existing data)
            # self.collection.drop()
            
            # Insert products in batches to avoid memory issues
            batch_size = 1000
            total_inserted = 0
            
            for i in range(0, len(products), batch_size):
                batch = products[i:i + batch_size]
                try:
                    result = self.collection.insert_many(batch, ordered=False)
                    total_inserted += len(result.inserted_ids)
                    logger.info(f"Inserted batch {i//batch_size + 1}: {len(result.inserted_ids)} products")
                except Exception as batch_error:
                    logger.warning(f"Batch {i//batch_size + 1} had some errors: {batch_error}")
                    # Continue with next batch
                    continue
            
            logger.info(f"Successfully migrated {total_inserted} products to MongoDB")
            return total_inserted > 0
                
        except Exception as e:
            logger.error(f"Failed to migrate CSV to MongoDB: {e}")
            return False
    
    def get_all_products(self) -> List[Dict[str, Any]]:
        """Get all products from MongoDB"""
        try:
            products = list(self.collection.find({}))
            logger.info(f"Retrieved {len(products)} products from MongoDB")
            return products
        except Exception as e:
            logger.error(f"Failed to retrieve products from MongoDB: {e}")
            return []
    
    def search_products(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search products by text query"""
        try:
            # Create regex-based search across fields present in our dataset
            search_query = {
                "$or": [
                    {"productDisplayName": {"$regex": query, "$options": "i"}},
                    {"articleType": {"$regex": query, "$options": "i"}},
                    {"usage": {"$regex": query, "$options": "i"}},
                    {"baseColour": {"$regex": query, "$options": "i"}},
                    {"gender": {"$regex": query, "$options": "i"}},
                    {"brand": {"$regex": query, "$options": "i"}},
                    {"masterCategory": {"$regex": query, "$options": "i"}},
                    {"subCategory": {"$regex": query, "$options": "i"}}
                ]
            }
            
            products = list(self.collection.find(search_query).limit(limit))
            logger.info(f"Found {len(products)} products matching query: {query}")
            return products
            
        except Exception as e:
            logger.error(f"Failed to search products: {e}")
            return []

    def search_products_fast(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Fast text search using MongoDB text index with score and projection.

        Falls back to regex search if text index is unavailable.
        """
        try:
            pipeline = [
                {"$match": {"$text": {"$search": query}}},
                {"$addFields": {"score": {"$meta": "textScore"}}},
                {"$sort": {"score": -1}},
                {"$project": {
                    "productDisplayName": 1,
                    "articleType": 1,
                    "usage": 1,
                    "baseColour": 1,
                    "gender": 1,
                    "brand": 1,
                    "masterCategory": 1,
                    "subCategory": 1,
                    "filename": 1,
                    "link": 1,
                    "score": 1
                }},
                {"$limit": int(limit)}
            ]
            cursor = self.collection.aggregate(pipeline, allowDiskUse=False, maxTimeMS=2000)
            products = list(cursor)
            return products
        except Exception as e:
            logger.warning(f"Text search failed or no index, falling back to regex: {e}")
            return self.search_products(query, limit)
    
    def get_products_by_category(self, category: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get products by category"""
        try:
            products = list(self.collection.find({"category": {"$regex": category, "$options": "i"}}).limit(limit))
            logger.info(f"Found {len(products)} products in category: {category}")
            return products
        except Exception as e:
            logger.error(f"Failed to get products by category: {e}")
            return []
    
    def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific product by ID"""
        try:
            from bson import ObjectId
            product = self.collection.find_one({"_id": ObjectId(product_id)})
            if product:
                logger.info(f"Retrieved product: {product.get('name', 'Unknown')}")
            return product
        except Exception as e:
            logger.error(f"Failed to get product by ID: {e}")
            return None

    def add_product(self, product: Dict[str, Any]) -> Optional[str]:
        """Insert a single product document and return its ID"""
        try:
            product = dict(product or {})
            product.setdefault('created_at', datetime.utcnow())
            product.setdefault('updated_at', datetime.utcnow())
            # Normalize field names similar to CSV schema
            # Expected keys: productDisplayName, articleType, usage, baseColour, gender, masterCategory, subCategory, brand, filename, link
            result = self.collection.insert_one(product)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to add product: {e}")
            return None
    
    def get_categories(self) -> List[str]:
        """Get all unique categories"""
        try:
            # Use the field name that exists in the migrated dataset
            # Original CSV uses 'masterCategory'
            categories = self.collection.distinct("masterCategory")
            logger.info(f"Found {len(categories)} unique categories")
            return categories
        except Exception as e:
            logger.error(f"Failed to get categories: {e}")
            return []
    
    def get_brands(self) -> List[str]:
        """Get all unique brands"""
        try:
            brands = self.collection.distinct("brand")
            logger.info(f"Found {len(brands)} unique brands")
            return brands
        except Exception as e:
            logger.error(f"Failed to get brands: {e}")
            return []
    
    def get_product_count(self) -> int:
        """Get total number of products"""
        try:
            count = self.collection.count_documents({})
            logger.info(f"Total products in database: {count}")
            return count
        except Exception as e:
            logger.error(f"Failed to get product count: {e}")
            return 0
    
    def create_text_index(self):
        """Create text index for better search performance"""
        try:
            # Create compound text index on fields we query
            self.collection.create_index([
                ("productDisplayName", "text"),
                ("articleType", "text"),
                ("usage", "text"),
                ("baseColour", "text"),
                ("gender", "text"),
                ("brand", "text"),
                ("masterCategory", "text"),
                ("subCategory", "text")
            ])
            logger.info("Created text index for product search")
        except Exception as e:
            logger.error(f"Failed to create text index: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            if self.collection is None:
                return {
                    "total_products": 0,
                    "categories": 0,
                    "subcategories": 0,
                    "genders": 0,
                    "brands": 0,
                    "database_name": self.database_name,
                    "collection_name": self.collection_name,
                    "connection_status": "disconnected"
                }
            
            stats = {
                "total_products": self.get_product_count(),
                "categories": len(self.get_categories()),
                "subcategories": len(self.collection.distinct("subCategory")),
                "genders": len(self.collection.distinct("gender")),
                "brands": len(self.get_brands()),
                "database_name": self.database_name,
                "collection_name": self.collection_name,
                "connection_status": "connected" if self.client else "disconnected"
            }
            return stats
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {"error": str(e)}

# Initialize MongoDB Manager
mongodb_manager = MongoDBManager()

def get_mongodb_manager() -> MongoDBManager:
    """Get MongoDB manager instance"""
    return mongodb_manager

def connect_to_mongodb() -> bool:
    """Connect to MongoDB and return connection status"""
    return mongodb_manager.connect()

def migrate_csv_data(csv_file_path: str) -> bool:
    """Migrate CSV data to MongoDB"""
    if not mongodb_manager.client:
        if not mongodb_manager.connect():
            return False
    return mongodb_manager.migrate_csv_to_mongodb(csv_file_path)

def get_products_from_mongodb(query: str = "", limit: int = 10) -> List[Dict[str, Any]]:
    """Get products from MongoDB with optional search query"""
    if not mongodb_manager.client:
        if not mongodb_manager.connect():
            return []
    
    if query:
        return mongodb_manager.search_products(query, limit)
    else:
        return mongodb_manager.get_all_products()[:limit]

def get_database_connection_status() -> Dict[str, Any]:
    """Get database connection status and statistics"""
    if not mongodb_manager.client:
        mongodb_manager.connect()
    
    return mongodb_manager.get_database_stats()
